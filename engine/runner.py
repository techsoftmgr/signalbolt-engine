"""
Multi-strategy signal scanner.

Five strategies each run on their own APScheduler job:
  scalping      — 5 min, tight momentum signals
  day_trade     — 15 min, intraday SMC signals
  swing_trade   — 60 min, higher-timeframe trend signals
  options_flow  — 15 min, unusual options activity (Pro+ only)
  dark_pool     — 15 min, large block trade detection (Pro+ only)

SQL needed before first run:
  ALTER TABLE signals
  ADD COLUMN IF NOT EXISTS strategy_type VARCHAR(20) DEFAULT 'day_trade';
"""

import logging
import os
import time
import sentry_sdk
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from supabase import create_client, Client

from engine import smc, scorer, explainer, options_scanner, push
from engine.tracker import track_signals
from engine import regime_detector, session_classifier, gamma_engine, manipulation_detector, sl_tp_engine, risk_manager
from engine import weight_optimizer, signal_monitor
from engine import unusual_whales as uw

load_dotenv()

logger = logging.getLogger("signalbolt.runner")

# ── Alpaca client singleton for runner price fetches ──────────
# Used by _analyze_dark_pool and _analyze_options_flow to get current price.
# Fix #4: was re-creating a new client per ticker — now shared across all calls.
_alpaca_runner_client = None
_alpaca_runner_ok     = False
try:
    from alpaca.data.historical import StockHistoricalDataClient as _SHDClient
    from alpaca.data.requests import StockLatestTradeRequest as _SLTRequest
    _alpaca_api_key    = os.environ.get("ALPACA_API_KEY", "")
    _alpaca_secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if _alpaca_api_key and _alpaca_secret_key:
        _alpaca_runner_client = _SHDClient(_alpaca_api_key, _alpaca_secret_key)
        _alpaca_runner_ok     = True
except Exception as _e:
    logger.debug(f"[runner] Alpaca client init failed: {_e}")

# ── Regime cache — shared across all strategy scans in a cycle ──
# Fix #7: regime_detector.detect() was called fresh for each of the 4 strategies.
# Each call = ~4 yfinance requests. Now cached 4 minutes (same TTL as stream.py).
_runner_regime_cache: tuple[Optional[dict], float] = (None, 0.0)
_RUNNER_REGIME_TTL = 240  # 4 minutes

# ── News cache: {ticker: (has_news: bool, fetched_at: float)} ─────────────
_news_cache: dict[str, tuple[bool, float]] = {}
_NEWS_CACHE_TTL = 900  # 15 minutes


def _has_recent_news(ticker: str, lookback_minutes: int = 60) -> bool:
    """
    Check Alpaca News API for breaking news on a ticker in the last N minutes.
    Results are cached for 15 min to avoid hammering the API per-ticker per-scan.
    Falls back to False if keys are missing or request fails.
    """
    now = time.monotonic()
    cached = _news_cache.get(ticker)
    if cached and (now - cached[1]) < _NEWS_CACHE_TTL:
        return cached[0]

    try:
        api_key    = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not api_secret:
            return False

        start_iso = (
            datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        resp = requests.get(
            "https://data.alpaca.markets/v1beta1/news",
            params={"symbols": ticker, "start": start_iso, "limit": 5},
            headers={
                "APCA-API-KEY-ID":     api_key,
                "APCA-API-SECRET-KEY": api_secret,
            },
            timeout=5,
        )
        result = resp.status_code == 200 and len(resp.json().get("news", [])) > 0
    except Exception as e:
        logger.debug(f"[runner] News check failed for {ticker}: {e}")
        result = False

    _news_cache[ticker] = (result, now)
    if result:
        logger.info(f"[runner] 📰 Breaking news detected for {ticker}")
    return result

# ---------------------------------------------------------------------------
# Ticker lists
# ---------------------------------------------------------------------------

SCALP_TICKERS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMD"]

ALL_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "TSLA", "AMD", "SPY", "QQQ", "IWM", "DIA",
    "COIN", "PLTR", "MSTR", "HOOD", "RBLX",
    "UBER", "ABNB", "JPM", "GS", "XOM", "CVX",
    "MARA", "RIOT", "CLSK", "MRNA", "BNTX",
]

# ---------------------------------------------------------------------------
# Strategy configs
# ---------------------------------------------------------------------------

STRATEGY_CONFIGS = [
    # NOTE: scalping is intentionally excluded here.
    # It is handled by engine/stream.py via Alpaca WebSocket bar events,
    # which fires within 2-5 seconds of each 5-min bar close (real-time).
    # APScheduler polling for scalping would add up to 5 min lag — not acceptable.
    {
        "type":              "day_trade",
        "tickers":           ALL_TICKERS,
        "interval":          "15m",
        "period":            "5d",
        "run_every_minutes": 15,
    },
    {
        "type":              "swing_trade",
        "tickers":           ALL_TICKERS,
        "interval":          "1h",
        "period":            "60d",
        "run_every_minutes": 60,
    },
    {
        "type":              "options_flow",
        "tickers":           ALL_TICKERS,
        "interval":          "15m",
        "period":            "5d",
        "run_every_minutes": 15,
    },
    {
        "type":              "dark_pool",
        "tickers":           ALL_TICKERS,
        "interval":          "15m",
        "period":            "5d",
        "run_every_minutes": 15,
    },
]

# Max hold time per strategy before auto-expiry
STRATEGY_MAX_HOLD_HOURS = {
    "scalping":     0.5,    # 30 minutes
    "day_trade":    24.0,
    "swing_trade":  240.0,  # 10 days
    "options_flow": 24.0,
    "dark_pool":    24.0,
}


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def _supabase_key() -> str:
    return os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]


def _supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], _supabase_key())


def _has_active_signal(sb: Client, ticker: str, strategy_type: str) -> bool:
    """Return True if an active signal already exists for this ticker+strategy combo."""
    try:
        result = (
            sb.table("signals")
            .select("id")
            .eq("ticker", ticker)
            .eq("status", "active")
            .eq("strategy_type", strategy_type)
            .execute()
        )
        return len(result.data) > 0
    except Exception as e:
        logger.error(f"[runner] Active-signal check failed for {ticker}/{strategy_type}: {e}")
        return False


def _write_signal(sb: Client, row: dict) -> None:
    try:
        sb.table("signals").insert(row).execute()
        logger.info(
            f"[runner] SIGNAL SAVED  {row['ticker']:6s} {row['direction']:5s} "
            f"[{row.get('strategy_type','?')}]  entry={row['entry_price']}  score={row['confidence_score']}"
        )
    except Exception as e:
        logger.error(f"[runner] Supabase insert failed for {row['ticker']}: {e}")


def _has_active_option_signal(sb: Client, ticker: str) -> bool:
    try:
        result = (
            sb.table("option_signals")
            .select("id")
            .eq("ticker", ticker)
            .eq("status", "active")
            .execute()
        )
        return len(result.data) > 0
    except Exception as e:
        logger.error(f"[runner] Option signal check failed for {ticker}: {e}")
        return False


def _write_option_signal(sb: Client, row: dict) -> None:
    try:
        sb.table("option_signals").insert(row).execute()
        logger.info(
            f"[runner] OPTION SAVED  {row['ticker']:6s} {row['contract_type']:4s} "
            f"strike={row['strike_price']}  exp={row['expiry_date']}  score={row['confidence_score']}"
        )
    except Exception as e:
        logger.error(f"[runner] Option signal insert failed for {row['ticker']}: {e}")


# ---------------------------------------------------------------------------
# Dark pool detection
# ---------------------------------------------------------------------------

def _analyze_dark_pool(ticker: str, interval: str = "15m", period: str = "5d") -> Optional[dict]:
    """
    Detect institutional dark pool / off-exchange block prints via Unusual Whales (primary)
    or volume spike > 3× average (fallback when UW key not set).
    """
    # ── Unusual Whales primary path ───────────────────────────
    from engine.config import UNUSUAL_WHALES_API_KEY
    if UNUSUAL_WHALES_API_KEY:
        try:
            prints = uw.fetch_dark_pool(ticker, min_size=50_000)
            if not prints:
                return None

            direction = uw.get_pool_direction(prints)
            if not direction:
                return None

            # Fix #4: use module-level singleton — no new client per ticker
            if _alpaca_runner_ok and _alpaca_runner_client is not None:
                trade = _alpaca_runner_client.get_stock_latest_trade(
                    _SLTRequest(symbol_or_symbols=ticker)
                )
                price = float(trade[ticker].price)
            else:
                import yfinance as yf
                price = float(yf.Ticker(ticker).fast_info.last_price or 0)

            total_notional = sum(p["notional"] for p in prints)
            largest_print  = max(prints, key=lambda p: p["notional"])

            logger.info(
                f"[runner] {ticker} UW dark pool: {len(prints)} prints "
                f"total=${total_notional:,.0f} "
                f"largest=${largest_print['notional']:,.0f} @ ${largest_print['price']:.2f} "
                f"→ {direction}"
            )

            df = smc.fetch_candles(ticker, period=period, interval=interval)

            if direction == "LONG":
                entry, stop_loss = round(price, 4), round(price * 0.992, 4)
                target_one, target_two = round(price * 1.015, 4), round(price * 1.030, 4)
            else:
                entry, stop_loss = round(price, 4), round(price * 1.008, 4)
                target_one, target_two = round(price * 0.985, 4), round(price * 0.970, 4)

            return {
                "ticker":          ticker,
                "current_price":   price,
                "direction":       direction,
                "entry":           entry,
                "stop_loss":       stop_loss,
                "target_one":      target_one,
                "target_two":      target_two,
                "total_notional":  total_notional,
                "print_count":     len(prints),
                "largest_print":   largest_print["notional"],
                "candles":         df,
                "timeframe":       interval,
                "strategy_type":   "dark_pool",
                "structure": {}, "fvgs": {}, "obs": {},
                "confidence_factors": [
                    f"Dark Pool Block — ${total_notional:,.0f} off-exchange",
                    f"{len(prints)} FINRA TRF print(s) detected",
                    f"Largest block: ${largest_print['notional']:,.0f} @ ${largest_print['price']:.2f}",
                ],
            }
        except Exception as e:
            logger.warning(f"[runner] UW dark pool failed for {ticker}: {e} — falling back")

    # ── Volume spike fallback (when UW key not yet set) ──────
    try:
        df = smc.fetch_candles(ticker, period=period, interval=interval)
        if df.empty or len(df) < 10:
            return None

        avg_volume   = float(df["volume"].iloc[:-5].mean())
        last_volume  = float(df["volume"].iloc[-1])
        if avg_volume <= 0 or last_volume < avg_volume * 3:
            return None

        last         = df.iloc[-1]
        price        = float(last["close"])
        direction    = "LONG" if float(last["close"]) >= float(last["open"]) else "SHORT"
        volume_ratio = last_volume / avg_volume

        if direction == "LONG":
            entry, stop_loss = round(price, 4), round(price * 0.992, 4)
            target_one, target_two = round(price * 1.015, 4), round(price * 1.030, 4)
        else:
            entry, stop_loss = round(price, 4), round(price * 1.008, 4)
            target_one, target_two = round(price * 0.985, 4), round(price * 0.970, 4)

        return {
            "ticker":        ticker,   "current_price": price,
            "direction":     direction, "entry":         entry,
            "stop_loss":     stop_loss, "target_one":    target_one,
            "target_two":    target_two, "volume_ratio": volume_ratio,
            "candles":       df,        "timeframe":     interval,
            "strategy_type": "dark_pool",
            "structure": {}, "fvgs": {}, "obs": {},
        }
    except Exception as e:
        logger.debug(f"[runner] Dark pool fallback failed for {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Options flow detection
# ---------------------------------------------------------------------------

def _analyze_options_flow(ticker: str) -> Optional[dict]:
    """
    Detect unusual options activity via Unusual Whales API (primary)
    or yfinance volume/OI ratio (fallback when UW key not set).
    """
    # ── Unusual Whales primary path ───────────────────────────
    from engine.config import UNUSUAL_WHALES_API_KEY
    if UNUSUAL_WHALES_API_KEY:
        try:
            flows = uw.fetch_options_flow(ticker, min_premium=100_000)
            if not flows:
                return None

            direction = uw.get_flow_direction(flows)
            if not direction:
                return None

            # Fix #4: use module-level singleton — no new client per ticker
            if _alpaca_runner_ok and _alpaca_runner_client is not None:
                trade = _alpaca_runner_client.get_stock_latest_trade(
                    _SLTRequest(symbol_or_symbols=ticker)
                )
                price = float(trade[ticker].price)
            else:
                import yfinance as yf
                price = float(yf.Ticker(ticker).fast_info.last_price or 0)

            # Aggregate flow stats for logging / confidence factors
            bull = [f for f in flows if f["sentiment"] == "bullish"]
            bear = [f for f in flows if f["sentiment"] == "bearish"]
            total_premium = sum(f["premium"] for f in flows)
            bull_premium  = sum(f["premium"] for f in bull)
            bear_premium  = sum(f["premium"] for f in bear)
            sweep_count   = sum(1 for f in flows if f.get("is_sweep"))
            block_count   = sum(1 for f in flows if f.get("is_floor"))

            logger.info(
                f"[runner] {ticker} UW options flow: {len(flows)} events "
                f"bull={len(bull)} bear={len(bear)} "
                f"sweeps={sweep_count} blocks={block_count} "
                f"total_premium=${total_premium:,.0f} → {direction}"
            )

            df = smc.fetch_candles(ticker, period="5d", interval="15m")

            if direction == "LONG":
                entry      = round(price, 4)
                stop_loss  = round(price * 0.992, 4)
                target_one = round(price * 1.015, 4)
                target_two = round(price * 1.030, 4)
            else:
                entry      = round(price, 4)
                stop_loss  = round(price * 1.008, 4)
                target_one = round(price * 0.985, 4)
                target_two = round(price * 0.970, 4)

            return {
                "ticker":          ticker,
                "current_price":   price,
                "direction":       direction,
                "entry":           entry,
                "stop_loss":       stop_loss,
                "target_one":      target_one,
                "target_two":      target_two,
                "call_volume":     sum(f["volume"] for f in bull),
                "put_volume":      sum(f["volume"] for f in bear),
                "total_premium":   total_premium,
                # Fix #1: bull/bear premium passed to scorer for L3 flow sentiment
                "bull_premium":    bull_premium,
                "bear_premium":    bear_premium,
                "sweep_count":     sweep_count,
                "block_count":     block_count,
                "candles":         df,
                "timeframe":       "15m",
                "strategy_type":   "options_flow",
                "structure": {}, "fvgs": {}, "obs": {},
                "confidence_factors": [
                    f"Unusual Options Flow — ${total_premium:,.0f} premium",
                    f"{sweep_count} sweep(s) + {block_count} block(s) detected",
                    f"Opening flow: {'bullish' if direction == 'LONG' else 'bearish'} dominance",
                ],
            }
        except Exception as e:
            logger.warning(f"[runner] UW options flow failed for {ticker}: {e} — falling back")

    # ── yfinance fallback (when UW key not yet set) ──────────
    try:
        import yfinance as yf
        tk    = yf.Ticker(ticker)
        price = tk.fast_info.last_price
        if not price:
            return None

        exps = tk.options
        if not exps:
            return None

        chain = tk.option_chain(exps[0])

        def unusual(df_opts: pd.DataFrame) -> int:
            atm     = df_opts[abs(df_opts["strike"] - price) / price < 0.05]
            valid   = atm[(atm["openInterest"] > 0) & (atm["volume"] > 0)]
            flagged = valid[valid["volume"] / valid["openInterest"] > 3]
            return int(flagged["volume"].sum()) if not flagged.empty else 0

        call_vol = unusual(chain.calls)
        put_vol  = unusual(chain.puts)
        if call_vol == 0 and put_vol == 0:
            return None

        direction = "LONG" if call_vol >= put_vol else "SHORT"
        df = smc.fetch_candles(ticker, period="5d", interval="15m")

        if direction == "LONG":
            entry, stop_loss = round(price, 4), round(price * 0.992, 4)
            target_one, target_two = round(price * 1.015, 4), round(price * 1.030, 4)
        else:
            entry, stop_loss = round(price, 4), round(price * 1.008, 4)
            target_one, target_two = round(price * 0.985, 4), round(price * 0.970, 4)

        return {
            "ticker":        ticker,   "current_price": price,
            "direction":     direction, "entry":         entry,
            "stop_loss":     stop_loss, "target_one":    target_one,
            "target_two":    target_two, "call_volume":  call_vol,
            "put_volume":    put_vol,   "candles":       df,
            "timeframe":     "15m",    "strategy_type": "options_flow",
            "structure": {}, "fvgs": {}, "obs": {},
        }
    except Exception as e:
        logger.debug(f"[runner] Options flow fallback failed for {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-ticker pipelines
# ---------------------------------------------------------------------------

def _process_smc_ticker(sb: Client, ticker: str, config: dict,
                        regime: dict = None, session: dict = None) -> None:
    """Standard SMC pipeline for scalping / day_trade / swing_trade — quant-upgraded."""
    strategy_type = config["type"]
    regime  = regime or {}
    session = session or {}

    if _has_active_signal(sb, ticker, strategy_type):
        logger.debug(f"[runner] {ticker} [{strategy_type}]: active signal exists — skipping")
        return

    analysis = smc.analyze(
        ticker,
        interval=config["interval"],
        period=config["period"],
        strategy_type=strategy_type,
    )
    if not analysis or not analysis.get("direction"):
        logger.debug(f"[runner] {ticker} [{strategy_type}]: no clear SMC direction")
        return

    direction = analysis["direction"]
    df        = analysis.get("candles")
    price     = analysis["current_price"]

    # ── QUANT GATE 3: Manipulation check ─────────────────────
    is_crypto = ticker in ("COIN", "MSTR", "MARA", "RIOT", "CLSK")
    has_news  = _has_recent_news(ticker)
    manipulation = manipulation_detector.detect(
        df, ticker, direction,
        has_news=has_news,
        is_crypto=is_crypto,
    )
    if manipulation_detector.is_blocking(manipulation):
        logger.info(f"[runner] {ticker} BLOCKED — manipulation: {manipulation['flags']}")
        return

    # ── QUANT GATE 4: Gamma exposure ──────────────────────────
    gamma = gamma_engine.fetch(ticker, price)

    # Score with quant layers
    scored = scorer.score(
        analysis, strategy_type,
        regime=regime,
        session=session,
        gamma=gamma,
        manipulation=manipulation,
    )
    sweep = analysis.get("liquidity_sweep", {})
    logger.info(
        f"[runner] {ticker} [{strategy_type}] score={scored['total']}/{scored['threshold']} "
        f"(L1={scored['breakdown']['l1_smc']} L2={scored['breakdown']['l2_technical']} "
        f"L3={scored['breakdown']['l3_sentiment']} L4={scored['breakdown']['l4_risk']} "
        f"L5={scored['breakdown'].get('l5_mtf', 0)} "
        f"L6={scored['breakdown'].get('l6_regime', 0)} "
        f"L7={scored['breakdown'].get('l7_session', 0)} "
        f"L8={scored['breakdown'].get('l8_gamma', 0)} "
        f"bonus={scored['breakdown'].get('quant_bonus', 0):+.1f})"
        + (f" SWEEP={sweep['candles_ago']}bars_ago" if sweep.get("swept") else "")
    )

    if not scored["passes"]:
        return

    # ── QUANT GATE 5: Gamma-aware SL/TP ──────────────────────
    sltp = sl_tp_engine.calculate(
        direction=direction,
        entry=price,
        df=df,
        regime=regime,
        session=session,
        gamma=gamma,
        strategy_type=strategy_type,
    )
    if not sltp["valid"]:
        logger.info(f"[runner] {ticker} BLOCKED — R:R={sltp['risk_reward_1']:.2f} < 2.0")
        return

    # Use quant SL/TP if better R:R, else keep SMC levels
    use_quant_sltp = sltp["risk_reward_1"] >= 2.0
    final_sl = sltp["stop_loss"] if use_quant_sltp else scored["stop_loss"]
    final_t1 = sltp["target_one"] if use_quant_sltp else scored["target_one"]
    final_t2 = sltp["target_two"] if use_quant_sltp else scored["target_two"]

    # ── QUANT GATE 6: Portfolio risk ──────────────────────────
    risk = risk_manager.check(sb, ticker, scored["total"])
    if not risk["allowed"]:
        logger.info(f"[runner] {ticker} BLOCKED — portfolio: {risk['block_reason']}")
        return

    # ── Swing trade: flag after-hours entry ──────────────────────
    # Swing signals are allowed to fire after market close (setup detected on
    # historical bars), but the entry price is the last bar's close — users
    # can't trade at that price until the next session opens.
    confidence_factors = list(scored.get("confidence_factors", []))
    if strategy_type == "swing_trade" and not session.get("market_open", True):
        confidence_factors.append(
            "⚠️ After-hours signal — enter at next session open, not at listed price"
        )

    signal_row = {
        "ticker":             ticker,
        "direction":          direction,
        "entry_price":        round(price, 2),
        "stop_loss":          final_sl,
        "target_one":         final_t1,
        "target_two":         final_t2,
        "confidence_score":   scored["total"],
        "confidence_factors": confidence_factors,
        "timeframe":          config["interval"],
        "strategy_type":      strategy_type,
        "status":             "active",
        "ai_explanation":     None,
        # Quant metadata
        "regime_type":        regime.get("regime_type", ""),
        "session_mode":       session.get("mode", ""),
        "confidence_tier":    risk["confidence_tier"],
        "position_multiplier": risk["position_mult"],
        "gamma_net_gex":      gamma.get("net_gex", 0),
        "gamma_is_negative":  gamma.get("is_negative_gamma", False),
        "manipulation_clean": manipulation.get("is_clean", True),
        "manipulation_flags": manipulation.get("flags", []),
        "sl_adjustments":     sltp.get("adjustments", []),
        "risk_reward":        sltp["risk_reward_1"],
        # Score breakdown stored for optimizer feedback loop
        "score_breakdown":    scored.get("breakdown", {}),
    }
    signal_row["ai_explanation"] = explainer.generate(signal_row, scored["breakdown"])
    _write_signal(sb, signal_row)

    try:
        push.send_signal_alert(ticker, direction, scored["total"], "stock")
    except Exception as e:
        logger.warning(f"[runner] Push notification failed for {ticker}: {e}")

    # Also scan options chain for day_trade and swing_trade (not scalping — too fast)
    if strategy_type in ("day_trade", "swing_trade") and not _has_active_option_signal(sb, ticker):
        opt = options_scanner.scan(
            ticker, direction,
            price,
            stock_target_one=final_t1,
        )
        if opt:
            opt["confidence_score"]   = scored["total"]
            opt["confidence_factors"] = scored.get("confidence_factors", [])
            opt["ai_explanation"]     = signal_row["ai_explanation"]
            opt["timeframe"]          = config["interval"]
            opt["strategy_type"]      = strategy_type
            opt["status"]             = "active"
            _write_option_signal(sb, opt)
            try:
                push.send_signal_alert(ticker, direction, scored["total"], "option")
            except Exception as e:
                logger.warning(f"[runner] Push notification failed for option {ticker}: {e}")


def _process_dark_pool_ticker(sb: Client, ticker: str, config: dict,
                              regime: dict = None, session: dict = None) -> None:
    regime  = regime  or {}
    session = session or {}

    if _has_active_signal(sb, ticker, "dark_pool"):
        return

    analysis = _analyze_dark_pool(ticker, config["interval"], config["period"])
    if not analysis:
        return

    direction = analysis["direction"]
    df        = analysis.get("candles")
    price     = analysis["current_price"]

    # ── Manipulation check ───────────────────────────────────
    is_crypto = ticker in ("COIN", "MSTR", "MARA", "RIOT", "CLSK")
    has_news  = _has_recent_news(ticker)
    manipulation = manipulation_detector.detect(
        df, ticker, direction, has_news=has_news, is_crypto=is_crypto,
    )
    if manipulation_detector.is_blocking(manipulation):
        logger.info(f"[runner] {ticker} [dark_pool] BLOCKED — manipulation: {manipulation['flags']}")
        return

    # ── Gamma exposure ────────────────────────────────────────
    gamma = gamma_engine.fetch(ticker, price)

    # ── Score with all quant layers ───────────────────────────
    scored = scorer.score(
        analysis, "dark_pool",
        regime=regime, session=session,
        gamma=gamma, manipulation=manipulation,
    )
    logger.info(
        f"[runner] {ticker} [dark_pool] score={scored['total']}/{scored['threshold']} "
        f"vol_ratio={analysis.get('volume_ratio', 0):.1f}x"
    )

    if not scored["passes"]:
        return

    # ── Gamma-aware SL/TP ─────────────────────────────────────
    sltp = sl_tp_engine.calculate(
        direction=direction, entry=price, df=df,
        regime=regime, session=session, gamma=gamma,
        strategy_type="dark_pool",
    )
    if not sltp["valid"]:
        logger.info(f"[runner] {ticker} [dark_pool] BLOCKED — R:R={sltp['risk_reward_1']:.2f} < 2.0")
        return

    # ── Portfolio risk ────────────────────────────────────────
    risk = risk_manager.check(sb, ticker, scored["total"])
    if not risk["allowed"]:
        logger.info(f"[runner] {ticker} [dark_pool] BLOCKED — portfolio: {risk['block_reason']}")
        return

    signal_row = {
        "ticker":             ticker,
        "direction":          direction,
        "entry_price":        round(price, 2),
        "stop_loss":          sltp["stop_loss"],
        "target_one":         sltp["target_one"],
        "target_two":         sltp["target_two"],
        "confidence_score":   scored["total"],
        "confidence_factors": scored.get("confidence_factors", [
            "Dark Pool Block Trade",
            f"{analysis.get('volume_ratio', 0):.1f}x Volume Spike",
        ]),
        "timeframe":          config["interval"],
        "strategy_type":      "dark_pool",
        "status":             "active",
        "ai_explanation":     None,
        # Quant metadata
        "regime_type":        regime.get("regime_type", ""),
        "session_mode":       session.get("mode", ""),
        "confidence_tier":    risk["confidence_tier"],
        "position_multiplier": risk["position_mult"],
        "gamma_net_gex":      gamma.get("net_gex", 0),
        "gamma_is_negative":  gamma.get("is_negative_gamma", False),
        "manipulation_clean": manipulation.get("is_clean", True),
        "manipulation_flags": manipulation.get("flags", []),
        "sl_adjustments":     sltp.get("adjustments", []),
        "risk_reward":        sltp["risk_reward_1"],
        "score_breakdown":    scored.get("breakdown", {}),
    }
    signal_row["ai_explanation"] = explainer.generate(signal_row, scored["breakdown"])
    _write_signal(sb, signal_row)

    try:
        push.send_signal_alert(ticker, direction, scored["total"], "stock")
    except Exception as e:
        logger.warning(f"[runner] Push failed for dark_pool {ticker}: {e}")


def _process_options_flow_ticker(sb: Client, ticker: str, config: dict,
                                 regime: dict = None, session: dict = None) -> None:
    regime  = regime  or {}
    session = session or {}

    if _has_active_signal(sb, ticker, "options_flow"):
        return

    analysis = _analyze_options_flow(ticker)
    if not analysis:
        return

    direction = analysis["direction"]
    df        = analysis.get("candles")
    price     = analysis["current_price"]

    # ── Manipulation check ───────────────────────────────────
    is_crypto = ticker in ("COIN", "MSTR", "MARA", "RIOT", "CLSK")
    has_news  = _has_recent_news(ticker)
    manipulation = manipulation_detector.detect(
        df, ticker, direction, has_news=has_news, is_crypto=is_crypto,
    )
    if manipulation_detector.is_blocking(manipulation):
        logger.info(f"[runner] {ticker} [options_flow] BLOCKED — manipulation: {manipulation['flags']}")
        return

    # ── Gamma exposure ────────────────────────────────────────
    gamma = gamma_engine.fetch(ticker, price)

    # ── Score with all quant layers ───────────────────────────
    scored = scorer.score(
        analysis, "options_flow",
        regime=regime, session=session,
        gamma=gamma, manipulation=manipulation,
    )
    logger.info(
        f"[runner] {ticker} [options_flow] score={scored['total']}/{scored['threshold']} "
        f"calls={analysis.get('call_volume', 0)} puts={analysis.get('put_volume', 0)}"
    )

    if not scored["passes"]:
        return

    # ── Gamma-aware SL/TP ─────────────────────────────────────
    sltp = sl_tp_engine.calculate(
        direction=direction, entry=price, df=df,
        regime=regime, session=session, gamma=gamma,
        strategy_type="options_flow",
    )
    if not sltp["valid"]:
        logger.info(f"[runner] {ticker} [options_flow] BLOCKED — R:R={sltp['risk_reward_1']:.2f} < 2.0")
        return

    # ── Portfolio risk ────────────────────────────────────────
    risk = risk_manager.check(sb, ticker, scored["total"])
    if not risk["allowed"]:
        logger.info(f"[runner] {ticker} [options_flow] BLOCKED — portfolio: {risk['block_reason']}")
        return

    signal_row = {
        "ticker":             ticker,
        "direction":          direction,
        "entry_price":        round(price, 2),
        "stop_loss":          sltp["stop_loss"],
        "target_one":         sltp["target_one"],
        "target_two":         sltp["target_two"],
        "confidence_score":   scored["total"],
        "confidence_factors": scored.get("confidence_factors", [
            "Unusual Options Activity",
            "High Volume/OI Ratio",
        ]),
        "timeframe":          "15m",
        "strategy_type":      "options_flow",
        "status":             "active",
        "ai_explanation":     None,
        # Quant metadata
        "regime_type":        regime.get("regime_type", ""),
        "session_mode":       session.get("mode", ""),
        "confidence_tier":    risk["confidence_tier"],
        "position_multiplier": risk["position_mult"],
        "gamma_net_gex":      gamma.get("net_gex", 0),
        "gamma_is_negative":  gamma.get("is_negative_gamma", False),
        "manipulation_clean": manipulation.get("is_clean", True),
        "manipulation_flags": manipulation.get("flags", []),
        "sl_adjustments":     sltp.get("adjustments", []),
        "risk_reward":        sltp["risk_reward_1"],
        "score_breakdown":    scored.get("breakdown", {}),
    }
    signal_row["ai_explanation"] = explainer.generate(signal_row, scored["breakdown"])
    _write_signal(sb, signal_row)

    try:
        push.send_signal_alert(ticker, direction, scored["total"], "stock")
    except Exception as e:
        logger.warning(f"[runner] Push failed for options_flow {ticker}: {e}")


def _process_ticker(sb: Client, ticker: str, config: dict,
                    regime: dict = None, session: dict = None) -> None:
    strategy_type = config["type"]
    ctx = {"regime": regime or {}, "session": session or {}}
    if strategy_type == "dark_pool":
        _process_dark_pool_ticker(sb, ticker, config, **ctx)
    elif strategy_type == "options_flow":
        _process_options_flow_ticker(sb, ticker, config, **ctx)
    else:
        _process_smc_ticker(sb, ticker, config, **ctx)


# ---------------------------------------------------------------------------
# Auto-close logic
# ---------------------------------------------------------------------------

def _close_signals(sb: Client) -> None:
    """
    Close signals that hit target/stop or exceeded their max hold time.
    Scalping signals expire after 30 min; swing trade after 10 days.
    """
    import yfinance as yf

    now = datetime.now(timezone.utc)

    # ── Stock signals ──
    try:
        rows = sb.table("signals").select("*").eq("status", "active").execute().data
    except Exception as e:
        logger.error(f"[closer] fetch signals failed: {e}")
        rows = []

    for sig in rows:
        created      = datetime.fromisoformat(sig["created_at"].replace("Z", "+00:00"))
        strategy     = sig.get("strategy_type") or "day_trade"
        hold_hours   = STRATEGY_MAX_HOLD_HOURS.get(strategy, 48.0)
        cutoff       = now - timedelta(hours=hold_hours)
        reason: Optional[str] = None
        close_price: Optional[float] = None

        if created < cutoff:
            reason = "expired"
        else:
            try:
                p = yf.Ticker(sig["ticker"]).fast_info.last_price
                if p:
                    close_price = float(p)
                    if sig["direction"] == "LONG":
                        if close_price >= sig["target_two"]:  reason = "target_hit"
                        elif close_price <= sig["stop_loss"]: reason = "stop_hit"
                    else:
                        if close_price <= sig["target_two"]:  reason = "target_hit"
                        elif close_price >= sig["stop_loss"]: reason = "stop_hit"
            except Exception:
                pass

        if reason:
            update: dict = {
                "status":        "closed",
                "closed_reason": reason,
                "closed_at":     now.isoformat(),
            }
            if reason == "expired":
                update["result"] = "expired"
            elif close_price is not None:
                entry   = float(sig["entry_price"])
                is_long = sig["direction"] == "LONG"
                if reason == "target_hit":
                    update["result"]     = "win"
                    hit_t2 = (is_long and close_price >= sig["target_two"]) or \
                             (not is_long and close_price <= sig["target_two"])
                    update["hit_target"] = "t2" if hit_t2 else "t1"
                else:
                    update["result"]     = "loss"
                    update["hit_target"] = "sl"
                raw_pct = ((close_price - entry) / entry) * 100 if is_long \
                          else ((entry - close_price) / entry) * 100
                raw_pnl = (close_price - entry) if is_long else (entry - close_price)
                update["result_pct"] = round(raw_pct, 4)
                update["result_pnl"] = round(raw_pnl, 4)
            try:
                sb.table("signals").update(update).eq("id", sig["id"]).execute()
                logger.info(f"[closer] CLOSED stock {sig['ticker']} [{strategy}] ({reason})")
            except Exception as e:
                logger.error(f"[closer] close failed for {sig['ticker']}: {e}")

    # ── Option signals ──
    try:
        opt_rows = sb.table("option_signals").select("*").eq("status", "active").execute().data
    except Exception as e:
        logger.error(f"[closer] fetch option_signals failed: {e}")
        opt_rows = []

    option_cutoff = now - timedelta(hours=24)
    for sig in opt_rows:
        created = datetime.fromisoformat(sig["created_at"].replace("Z", "+00:00"))
        reason  = None

        if created < option_cutoff:
            reason = "expired"
        else:
            try:
                price      = yf.Ticker(sig["ticker"]).fast_info.last_price
                delta      = sig.get("delta") or 0
                und_price  = sig.get("underlying_price") or 0
                entry_prem = sig.get("entry_premium") or 0
                if price and und_price and delta:
                    est = entry_prem + delta * (price - und_price)
                    if est >= sig["target_premium"]: reason = "target_hit"
                    elif est <= sig["stop_premium"]: reason = "stop_hit"
            except Exception:
                pass

        if reason:
            try:
                opt_result = "expired" if reason == "expired" else ("win" if reason == "target_hit" else "loss")
                sb.table("option_signals").update({
                    "status":        "closed",
                    "closed_reason": reason,
                    "closed_at":     now.isoformat(),
                    "result":        opt_result,
                }).eq("id", sig["id"]).execute()
                logger.info(f"[closer] CLOSED option {sig['ticker']} {sig['contract_type']} ({reason})")
            except Exception as e:
                logger.error(f"[closer] option close failed for {sig['ticker']}: {e}")


# ---------------------------------------------------------------------------
# Strategy scan jobs
# ---------------------------------------------------------------------------

def _run_strategy_scan(config: dict) -> None:
    strategy_type = config["type"]
    tickers       = config["tickers"]
    logger.info(
        f"[runner] {strategy_type.upper()} scan started — "
        f"{len(tickers)} tickers @ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
    )

    # ── QUANT GATE 1: Fetch market-wide context (cached 4 min across strategies) ──
    # Fix #7: all 4 strategy scans share one regime result per 4-minute window
    # instead of each making fresh yfinance calls.
    global _runner_regime_cache
    _cached_regime, _cached_regime_ts = _runner_regime_cache
    if _cached_regime is None or (time.monotonic() - _cached_regime_ts) > _RUNNER_REGIME_TTL:
        try:
            regime = regime_detector.detect()
            _runner_regime_cache = (regime, time.monotonic())
            logger.debug(f"[runner] Regime refreshed: {regime['regime_type']} VIX={regime['vix']:.1f}")
        except Exception as e:
            logger.warning(f"[runner] Regime detection failed: {e} — using neutral")
            regime = {"regime_type": "RANGING", "vix": 18.0, "vix_change_pct": 0.0,
                      "above_200ma": True, "adx": 20.0, "blocked": False, "block_reason": ""}
            _runner_regime_cache = (regime, time.monotonic())
    else:
        regime = _cached_regime
        logger.debug(f"[runner] Regime cache hit: {regime['regime_type']}")

    # ── QUANT GATE 2: Classify session ────────────────────────
    try:
        session = session_classifier.classify()
    except Exception as e:
        logger.warning(f"[runner] Session classification failed: {e}")
        session = {"mode": "STANDARD", "market_open": True, "blocked": False,
                   "block_reason": "", "threshold": 70, "sl_adjustment": 1.0,
                   "allows_swing": True, "is_opex_day": False, "is_opex_week": False}

    # ── Block entire scan if market closed / FOMC / PANIC ─────
    if session.get("blocked"):
        logger.info(f"[runner] {strategy_type.upper()} scan SKIPPED — {session['block_reason']}")
        return

    if not session.get("market_open") and strategy_type != "swing_trade":
        logger.info(f"[runner] {strategy_type.upper()} scan SKIPPED — market closed")
        return

    if regime.get("blocked") and strategy_type in ("scalping", "day_trade"):
        logger.info(f"[runner] {strategy_type.upper()} scan SKIPPED — {regime['block_reason']}")
        return

    # Block swing signals on OpEx day
    if strategy_type == "swing_trade" and not session.get("allows_swing"):
        logger.info(f"[runner] swing_trade scan SKIPPED — OpEx day (no swing signals)")
        return

    logger.info(
        f"[runner] Quant context: regime={regime['regime_type']} "
        f"VIX={regime['vix']:.1f} session={session['mode']} "
        f"threshold={session['threshold']}"
    )

    # ── Pre-fetch UW global feeds once (2 API calls cover all tickers) ──
    if strategy_type in ("options_flow", "dark_pool"):
        try:
            uw.warm_global_cache()
        except Exception as e:
            logger.warning(f"[runner] UW cache warm failed: {e} — will attempt per-ticker")

    sb = _supabase()
    fired = 0
    for ticker in tickers:
        try:
            _process_ticker(sb, ticker, config, regime=regime, session=session)
            fired += 1
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(f"[runner] Error processing {ticker} [{strategy_type}]: {e}", exc_info=True)
    logger.info(f"[runner] {strategy_type.upper()} scan complete — {fired}/{len(tickers)} processed")


def _run_maintenance() -> None:
    """Track open signal results and auto-close hits. Runs every 15 min."""
    logger.info("[runner] Maintenance started")
    try:
        track_signals()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[runner] track_signals failed: {e}")
    try:
        _close_signals(_supabase())
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[runner] _close_signals failed: {e}")
    try:
        signal_monitor.run()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[runner] signal_monitor failed: {e}")
    logger.info("[runner] Maintenance complete")


def _run_weight_optimization() -> None:
    """
    Weekly self-learning job: backtests history + tunes L1-L9 weights.
    Runs Sunday at 2 AM UTC. Takes ~5-15 min depending on data volume.
    New weights are immediately active for the next scan cycle.
    """
    logger.info("[runner] ═══ Weekly weight optimization job started ═══")
    try:
        summary = weight_optimizer.run_full_optimization()
        updated = sum(
            1 for strat in summary.values()
            for combo in strat.values()
            if isinstance(combo, dict) and combo.get("updated")
        )
        logger.info(f"[runner] ═══ Optimization complete — {updated} weight sets improved ═══")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[runner] Weight optimization failed: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def start_scheduler() -> BackgroundScheduler:
    """
    APScheduler now handles ONLY:
      1. Maintenance (tracker + signal_monitor) every 15 min
      2. Weekly weight optimization (Sunday 2 AM UTC)

    ALL strategy scans (scalping, day_trade, swing_trade, options_flow, dark_pool)
    are fired by stream.py at bar-boundary events — within 2-5 seconds of each
    bar close rather than on a fixed polling interval.
    """
    scheduler = BackgroundScheduler(timezone="UTC")
    now = datetime.now(timezone.utc)

    # ── Maintenance: tracker + signal_monitor every 15 min ───────────────
    scheduler.add_job(
        _run_maintenance,
        trigger=IntervalTrigger(minutes=15),
        id="maintenance",
        name="SignalBolt maintenance",
        replace_existing=True,
        next_run_time=now,
    )
    logger.info("[runner] Scheduled maintenance every 15 min")

    # ── EOD signal monitor: every 5 min from 2:55 PM to 4:05 PM ET ──────
    # Runs signal_monitor only — no strategy scans. Fires frequently enough
    # to catch the 3:00 PM warning window and the 3:30 PM force-close window
    # without the coarse 15-min gap of the main maintenance cycle.
    def _eod_monitor_job():
        from zoneinfo import ZoneInfo as _ZI
        from datetime import datetime as _dt
        _now = _dt.now(_ZI("America/New_York"))
        _mins = _now.hour * 60 + _now.minute
        # Only run during 2:55 PM (875) to 4:05 PM (965) ET window
        if 875 <= _mins <= 965 and _now.weekday() < 5:
            try:
                signal_monitor.run()
            except Exception as _e:
                logger.error(f"[runner] EOD monitor failed: {_e}")

    scheduler.add_job(
        _eod_monitor_job,
        trigger=IntervalTrigger(minutes=5),
        id="eod_monitor",
        name="EOD signal monitor (5-min near close)",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled EOD signal monitor every 5 min (active 2:55–4:05 PM ET)")

    # ── Weekly self-learning optimization (Sunday 2 AM UTC) ──────────────
    from apscheduler.triggers.cron import CronTrigger
    scheduler.add_job(
        _run_weight_optimization,
        trigger=CronTrigger(day_of_week="sun", hour=2, minute=0, timezone="UTC"),
        id="weight_optimization",
        name="SignalBolt weight optimizer",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled weekly weight optimization (Sunday 2:00 AM UTC)")

    scheduler.start()
    logger.info(
        "[runner] Scheduler started — "
        "signal scans are event-driven via stream.py WebSocket bar events"
    )
    return scheduler


def run_strategy_by_type(strategy_type: str) -> None:
    """
    Run a full strategy scan by name.
    Called by stream.py at bar-boundary events (event-driven).
    Also usable for manual runs / tests.
    """
    config = next((c for c in STRATEGY_CONFIGS if c["type"] == strategy_type), None)
    if config:
        _run_strategy_scan(config)
    else:
        logger.warning(f"[runner] Unknown strategy type: {strategy_type}")


# ---------------------------------------------------------------------------
# Legacy entry point (kept for backward compatibility with main.py /run endpoint)
# ---------------------------------------------------------------------------

def run_scan(tickers=None) -> None:
    """Run a single day_trade scan synchronously (used by /run endpoint and tests)."""
    config = next(c for c in STRATEGY_CONFIGS if c["type"] == "day_trade")
    if tickers:
        config = {**config, "tickers": tickers}
    _run_strategy_scan(config)
