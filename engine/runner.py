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
from engine import prescreener
from engine import chop_detector
from engine import mean_reversion
from engine import gap_engine
from engine import entry_gate
from engine import trade_tape
from engine import compression_detector, pullback_detector, swing_breakout_detector


def _tape_bonus(ticker: str) -> dict:
    """Safely compute the tape bonus — never let it break signal-write."""
    try:
        return trade_tape.compute_signal_bonus(ticker)
    except Exception:
        return {"bonus": 0, "reasons": ["tape error"]}
from engine import analytics as signal_analytics
from engine import premarket_scanner
from engine.setup_lifecycle import (
    SetupLifecycleManager,
    classify_setup_type,
    annotate_score,
)
_lifecycle = SetupLifecycleManager()

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

# Stop cooldown removed — the 9-layer scorer, chop detector, regime gate,
# and manipulation check are the re-entry filter. If structure has genuinely
# recovered after a stop hit the engine should take the trade. If it hasn't,
# the score won't pass threshold. Time-based lockouts override the engine's
# own intelligence and silently miss valid setups.


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

SCALP_TICKERS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMD", "MSFT", "META", "GOOGL", "COIN"]

# ALL_TICKERS is now DYNAMIC — populated by the pre-screener at scan time.
# The screener covers 500+ liquid US stocks and returns the ~50 showing
# the most momentum + volume activity at the moment of scanning.
# CORE_TICKERS are always included as a baseline.
# For backward-compat (used by /run endpoint), keep a static fallback list.
ALL_TICKERS = prescreener.CORE_TICKERS  # overridden dynamically in _run_strategy_scan

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
        "run_every_minutes": 10,   # APScheduler job fires every 10 min during market hours
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
    "scalping":       0.5,    # 30 minutes
    "day_trade":      8.0,    # intraday — closes within one session
    "vwap_reclaim":   8.0,    # mean-reversion intraday — same session
    "gap_fill":       8.0,    # gap/ORB play — resolves same session
    "pre_market":     8.0,    # pre-market breakout — resolves at/after open
    "swing_trade":    240.0,  # 10 days
    "earnings":       48.0,   # 2 days — pre/post earnings move
    "short_squeeze":  24.0,   # 1 day — squeeze resolves quickly
    "position_trade": 720.0,  # 30 days — macro position
    "options_flow":   8.0,
    "dark_pool":      8.0,
}


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def _supabase_key() -> str:
    return os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]


def _supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], _supabase_key())


def _has_active_signal(sb: Client, ticker: str, strategy_type: str) -> bool:
    """
    Return True if an active signal already exists for this ticker.

    Checks ALL strategies, not just the requested one — prevents the engine
    from piling multiple strategies (e.g. scalp + day_trade) onto the same
    ticker simultaneously when market conditions are bad for that ticker.
    Swing trades are exempt: they run on a different timeframe and can
    co-exist with an intraday signal on the same ticker.
    """
    try:
        query = (
            sb.table("signals")
            .select("id, strategy_type")
            .eq("ticker", ticker)
            .eq("status", "active")
        )
        # Swing co-exists with intraday (different timeframe/thesis).
        # But two intraday strategies (scalp + day_trade) on the same ticker
        # at the same moment is double-exposure on the same move — block it.
        if strategy_type != "swing_trade":
            query = query.neq("strategy_type", "swing_trade")

        result = query.execute()
        if result.data:
            existing = result.data[0].get("strategy_type", "?")
            logger.debug(
                f"[runner] {ticker} [{strategy_type}]: active {existing} signal exists — skipping"
            )
            return True
        return False
    except Exception as e:
        logger.error(f"[runner] Active-signal check failed for {ticker}/{strategy_type}: {e}")
        return False




def _ensure_stream_subscription(ticker: str, cap: int | None = None) -> None:
    """
    Add `ticker` to the live Alpaca trade stream so on_trade ticks → real-time
    SL/TP checks (_check_rt_levels) fire as soon as price crosses a level.

    Without this, only the fixed watchlist + prescreener movers have tick
    subscriptions; freshly-fired signals on other tickers had to wait for
    the 5-min signal_monitor to catch stop hits. User saw a 1% extra loss
    on AAL because of exactly this gap (stop crossed at 9:22, closed 9:27).

    `cap`: if set and the ticker isn't already subscribed, skip when the live
    subscription set has reached this size. Used by the predictive scan to bound
    growth from pre-screened movers. Active signals call with cap=None so their
    stops are always watched.

    Safe to call from sync context — schedules the async subscribe on the
    running event loop if there is one; queues otherwise.
    """
    try:
        import asyncio
        from engine import stream as _stream
        if cap is not None and ticker not in _stream._subscribed_tickers \
                and len(_stream._subscribed_tickers) >= cap:
            return
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            asyncio.ensure_future(_stream.subscribe_extra_tickers([ticker]))
        else:
            # No running loop (e.g. sync APScheduler thread). The function
            # itself queues to _pending_tickers when _wss_ref is None, so
            # just touch _subscribed_tickers directly.
            _stream._subscribed_tickers.add(ticker)
            _stream._pending_tickers.add(ticker)
    except Exception as e:
        logger.debug(f"[runner] stream subscribe failed for {ticker}: {e}")


def _write_signal(sb: Client, row: dict) -> str | None:
    """Insert signal row, log the 'fired' event, and return the new signal ID."""
    try:
        result = sb.table("signals").insert(row).execute()
        logger.info(
            f"[runner] SIGNAL SAVED  {row['ticker']:6s} {row['direction']:5s} "
            f"[{row.get('strategy_type','?')}]  entry={row['entry_price']}  score={row['confidence_score']}"
        )
        # CRITICAL: subscribe the ticker to the live trade stream so real-time
        # SL/TP checks fire on every tick instead of waiting 5 min for
        # signal_monitor. Without this, tight stops can blow past the level
        # without the engine reacting (see AAL incident 2026-05-27).
        _ensure_stream_subscription(row["ticker"])
        # NOTE: the 'fired' signal_event is created by the DB trigger
        # trg_signal_fired (see supabase-signal-fired-rich-note.sql) — which
        # now produces the rich note. We do NOT insert it here too, or we'd
        # get duplicate fired events (the bug fixed 2026-05-28).
        sig_id: str | None = None
        try:
            sig_id = result.data[0]["id"] if result.data else None
        except Exception:
            sig_id = None
        return sig_id   # caller uses this for push deep-link
    except Exception as e:
        # The partial unique index `idx_unique_active_signal` enforces
        # "one active signal per (ticker, strategy_type)" at the DB level
        # to close the TOCTOU race between APScheduler scans and
        # stream.py bar-event scans. When the race fires, the second
        # INSERT trips the index and Postgres raises 23505. That's the
        # ENGINE WORKING AS DESIGNED — log it at info, not error, so we
        # don't page on duplicate-scan benign collisions.
        err_str = str(e)
        if "23505" in err_str or "duplicate key value" in err_str or "idx_unique_active_signal" in err_str:
            logger.info(
                f"[runner] Duplicate active signal blocked at DB for "
                f"{row['ticker']}/{row.get('strategy_type','?')} "
                f"(race with concurrent scan — expected, no action)"
            )
        else:
            logger.error(f"[runner] Supabase insert failed for {row['ticker']}: {e}")
    return None


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


def _write_option_signal(sb: Client, row: dict) -> str | None:
    """Insert option signal row and return the new option signal ID."""
    try:
        result = sb.table("option_signals").insert(row).execute()
        logger.info(
            f"[runner] OPTION SAVED  {row['ticker']:6s} {row['contract_type']:4s} "
            f"strike={row['strike_price']}  exp={row['expiry_date']}  score={row['confidence_score']}"
        )
        return result.data[0]["id"] if result.data else None
    except Exception as e:
        # Same race protection as _write_signal — see comment there.
        err_str = str(e)
        if "23505" in err_str or "duplicate key value" in err_str or "idx_unique_active_option_signal" in err_str:
            logger.info(
                f"[runner] Duplicate active option signal blocked at DB for "
                f"{row['ticker']} (race with concurrent scan — expected, no action)"
            )
        else:
            logger.error(f"[runner] Option signal insert failed for {row['ticker']}: {e}")
    return None


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

def _process_mr_ticker(sb: Client, ticker: str, config: dict,
                       regime: dict = None, session: dict = None) -> bool:
    """
    Mean-reversion pipeline — only runs in RANGING / LOW_VOL regimes.
    Returns True if a signal fired (so caller can skip the SMC pipeline).
    """
    regime  = regime  or {}
    session = session or {}

    if _has_active_signal(sb, ticker, config["type"]):
        return False

    try:
        from engine import alpaca_client as _alpaca
        df    = _alpaca.get_bars(ticker, timeframe=config.get("interval", "15m"), days=5)
        price = _alpaca.get_latest_price(ticker)
        if df is None or df.empty or not price:
            return False

        mr = mean_reversion.analyze(
            ticker, df, price,
            regime=regime,
            interval=config.get("interval", "15m"),
        )
        if mr is None or not mr.passes:
            return False

        # ── Portfolio risk check ──────────────────────────────────
        risk = risk_manager.check(sb, ticker, mr.score)
        if not risk["allowed"]:
            logger.info(f"[runner] {ticker} [MR] BLOCKED — portfolio: {risk['block_reason']}")
            return False

        # ── Build signal row ──────────────────────────────────────
        signal_row = mean_reversion.to_signal_dict(mr, session)
        signal_row.update({
            "strategy_type":       "vwap_reclaim",   # always tagged as VWAP reclaim strategy
            "timeframe":           config.get("interval", "15m"),
            "regime_type":         regime.get("regime_type", ""),
            "session_mode":        session.get("mode", ""),
            "confidence_tier":     risk["confidence_tier"],
            "position_multiplier": risk["position_mult"],
            "setup_type":          "VWAP_MEAN_REVERSION",
            "confidence_grade":    "B+" if mr.score >= 74 else "B",
            "chop_score":          0.0,   # MR signals trade IN chop — no chop penalty
            "score_breakdown":     {"detector_source": "MEAN_REVERSION",
                                    "mr_score": mr.score, "mr_passes": list(mr.passes)},
        })
        signal_row["ai_explanation"] = explainer.generate(signal_row, signal_row["score_breakdown"])
        _write_signal(sb, signal_row)

        try:
            push.send_signal_alert(ticker, mr.direction, mr.score, "stock")
        except Exception as e:
            logger.warning(f"[runner] Push failed for MR {ticker}: {e}")

        logger.info(
            f"[runner] {ticker} [MR] FIRED — score={mr.score} dir={mr.direction} "
            f"entry={mr.entry:.2f} sl={mr.stop_loss:.2f} t1={mr.target_one:.2f}"
        )
        return True

    except Exception as e:
        logger.warning(f"[runner] MR pipeline failed for {ticker}: {e}")
        return False


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

    # ── Gap-ORB fallback when SMC finds no structure ──────────────────────────
    # On earnings gap days, the stock trades in "fresh air" — price levels with
    # no historical candles, so SMC has no OBs, FVGs, or BOS/CHoCH to work with.
    # The Opening Range (first 30 min) is the structural anchor on gap days.
    # Only attempt this for day_trade (15m bars) — not scalping or swing.
    # Signals from gap_engine are tagged "gap_fill" so they appear separately
    # in the analytics tab and get the correct max-hold window (same session).
    _from_gap_engine = False
    if (not analysis or not analysis.get("direction")) and strategy_type == "day_trade":
        try:
            from engine import alpaca_client as _alpaca
            _price = _alpaca.get_latest_price(ticker) or 0.0
            _df    = analysis.get("candles") if analysis else None
            if _df is None or _df.empty:
                _df = _alpaca.get_bars(ticker, timeframe="15m", days=5)
            if _df is not None and not _df.empty and _price > 0:
                gap_analysis = gap_engine.analyze(
                    ticker, _df, _price, strategy_type=strategy_type
                )
                if gap_analysis and gap_analysis.get("direction"):
                    analysis = gap_analysis
                    _from_gap_engine = True
                    logger.info(
                        f"[runner] {ticker} [gap_engine] SMC no-direction → "
                        f"GAP-ORB {gap_analysis['direction']} setup found "
                        f"(gap={gap_analysis.get('gap_pct', 0):+.1f}% "
                        f"score={gap_analysis.get('confidence_score', 0)})"
                    )
        except Exception as _ge:
            logger.debug(f"[runner] gap_engine fallback error for {ticker}: {_ge}")

    if not analysis or not analysis.get("direction"):
        logger.debug(f"[runner] {ticker} [{strategy_type}]: no clear direction (SMC+gap)")
        return

    direction = analysis["direction"]
    df        = analysis.get("candles")
    price     = analysis["current_price"]

    # ── QUANT GATE 2b: Chop detection ────────────────────────
    # Run BEFORE manipulation/gamma to avoid wasting those calls on choppy bars.
    regime_type_str = regime.get("regime_type", "UNKNOWN")
    chop = chop_detector.detect(
        df,
        regime_type=regime_type_str,
        interval=config.get("interval", "15m"),
        strategy_type=strategy_type,
    )
    is_gap_orb = analysis.get("setup_type") == "GAP_ORB"
    if chop.is_choppy:
        if is_gap_orb:
            # GAP_ORB setups are EXEMPT from chop blocking.
            # The tight ORB consolidation IS the setup — the market is
            # digesting the gap, not actually choppy.  Override is_choppy
            # so the scorer doesn't penalise it.
            chop.is_choppy = False
            logger.debug(f"[runner] {ticker} GAP_ORB chop gate bypassed — tight ORB expected")
        else:
            logger.info(
                f"[runner] {ticker} [{strategy_type}] SKIPPED — chop "
                f"score={chop.chop_score:.0f} > {chop.threshold_used:.0f} "
                f"({', '.join(chop.reasons[:2])})"
            )
            return

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

    # Score with quant layers + chop penalty
    scored = scorer.score(
        analysis, strategy_type,
        regime=regime,
        session=session,
        gamma=gamma,
        manipulation=manipulation,
        chop=chop,
    )
    sweep = analysis.get("liquidity_sweep", {})
    breakdown = scored.get("breakdown", {}) or {}
    logger.info(
        f"[runner] {ticker} [{strategy_type}] score={scored.get('total', 0)}/{scored.get('threshold', 0)} "
        f"grade={scored.get('confidence_grade','?')} "
        f"(L1={breakdown.get('l1_smc', 0)} L2={breakdown.get('l2_technical', 0)} "
        f"L3={breakdown.get('l3_sentiment', 0)} L4={breakdown.get('l4_risk', 0)} "
        f"L5={breakdown.get('l5_mtf', 0)} "
        f"L6={breakdown.get('l6_regime', 0)} "
        f"L7={breakdown.get('l7_session', 0)} "
        f"L8={breakdown.get('l8_gamma', 0)} "
        f"bonus={breakdown.get('quant_bonus', 0):+.1f} "
        f"chop_pen={breakdown.get('chop_penalty', 0):.1f})"
        + (f" SWEEP={sweep['candles_ago']}bars_ago" if sweep.get("swept") else "")
    )

    # ── Lifecycle stage gate ──────────────────────────────────
    # Scores below CONFIRMED_MIN (78) feed the WATCHLIST/DEVELOPING stages.
    # Only CONFIRMED scores proceed to SL/TP and signal write.
    try:
        sltp_preview = sl_tp_engine.calculate(
            direction=direction, entry=price, df=df,
            regime=regime, session=session, gamma=gamma,
            strategy_type=strategy_type,
            interval=config.get("interval", "15m"),
        )
        setup_type_str = classify_setup_type(analysis, session)
        _lifecycle.upsert_setup(
            analysis=analysis,
            score_result=scored,
            regime=regime,
            session=session,
            chop_result=chop,
            setup_type=setup_type_str,
            sltp=sltp_preview,
        )
    except Exception as _lc_err:
        logger.debug(f"[runner] lifecycle upsert skipped: {_lc_err}")
        setup_type_str = "UNKNOWN"

    if not scored["passes"]:
        return

    # ── ENTRY GATE v2: multi-timeframe + pattern confirmation ─
    # Runs AFTER scoring passes (cheap fast-path) and BEFORE expensive
    # SL/TP calc. Rejects signals that look good on the entry timeframe
    # but lack 15m trend agreement, 5m MACD lean, 1m reversal candle,
    # or print obvious bad-entry patterns (3 reds, overextended, vol drop).
    # See engine/entry_gate.py for details.
    entry_gate_log: dict = {}
    try:
        gate = entry_gate.check(
            ticker        = ticker,
            direction     = direction,
            strategy_type = strategy_type,
            df_entry      = df,
            price         = price,
            entry_tf      = config.get("interval", "15m"),
        )
        entry_gate_log = dict(gate.gate_log)
        if not gate.allowed:
            logger.info(
                f"[runner] {ticker} [{strategy_type}] BLOCKED — entry_gate: "
                + " | ".join(gate.reasons)
            )
            entry_gate.log_rejection(
                sb               = sb,
                ticker           = ticker,
                direction        = direction,
                strategy_type    = strategy_type,
                price            = price,
                confidence_score = scored.get("total", 0),
                gate             = gate,
            )
            return
        logger.debug(f"[runner] {ticker} entry_gate PASS — {gate.gate_log}")
    except Exception as _gate_err:
        # Fail open on any unexpected error — no worse than today's engine
        logger.warning(f"[runner] {ticker} entry_gate error (failing open): {_gate_err}")
        entry_gate_log = {"error": str(_gate_err)}

    # ── QUANT GATE 5: Realistic SL/TP ────────────────────────
    # interval is used to select the correct ATR method (H-L for intraday)
    # and to estimate the Average Daily Range for target capping.
    sltp = sl_tp_engine.calculate(
        direction=direction,
        entry=price,
        df=df,
        regime=regime,
        session=session,
        gamma=gamma,
        strategy_type=strategy_type,
        interval=config.get("interval", "15m"),
    )
    if not sltp["valid"]:
        logger.info(
            f"[runner] {ticker} BLOCKED — R:R={sltp['risk_reward_1']:.2f} below minimum "
            f"(atr={sltp['atr']:.3f} adr={sltp.get('adr', 0):.2f} strategy={strategy_type})"
        )
        return

    # SL/TP always comes from the quant engine — SMC fallback removed (dead code)
    final_sl = sltp["stop_loss"]
    final_t1 = sltp["target_one"]
    final_t2 = sltp["target_two"]

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

    # Compute tape bonus once so both confidence_score and breakdown use same value
    _tape_b = _tape_bonus(ticker)
    _adjusted_score = max(0, min(100, scored["total"] + _tape_b.get("bonus", 0)))
    if _tape_b.get("bonus", 0) != 0:
        confidence_factors = list(scored.get("confidence_factors", [])) + _tape_b.get("reasons", [])

    signal_row = {
        "ticker":             ticker,
        "direction":          direction,
        "entry_price":        round(price, 2),
        "stop_loss":          final_sl,
        "target_one":         final_t1,
        "target_two":         final_t2,
        "confidence_score":   _adjusted_score,
        "confidence_factors": confidence_factors,
        "timeframe":          config["interval"],
        # gap_engine signals get their own strategy tag so they appear separately
        # in analytics and use the correct hold window (STRATEGY_MAX_HOLD_HOURS)
        "strategy_type":      "gap_fill" if _from_gap_engine else strategy_type,
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
        # Score breakdown stored for optimizer feedback loop.
        # entry_gate_log + sl_width_pct + tape_bonus nested in here (no schema
        # change needed) so we can later correlate everything vs realized outcome.
        "score_breakdown":    {
            # Explicit tag so every signal carries the methodology that
            # produced it. Lets us A/B SMC vs predictive vs gap-engine vs MR
            # vs flow/dark_pool from a single DB query.
            "detector_source": "GAP_ENGINE" if _from_gap_engine else "SMC",
            **scored.get("breakdown", {}),
            "entry_gate":    entry_gate_log,
            "sl_width_pct":  round(abs(price - final_sl) / price * 100, 3) if price else None,
            "atr_used":      round(sltp.get("atr", 0), 4),
            "adr_used":      round(sltp.get("adr", 0), 4),
            # Tape bonus computed at fire-time from in-memory rolling tape.
            # Affects displayed confidence and feeds the optimizer feedback loop.
            "tape_bonus":    _tape_b,
        },
        # New lifecycle / quality metadata
        "confidence_grade":   scored.get("confidence_grade", "B"),
        "risk_grade":         scored.get("risk_grade", "MEDIUM"),
        "chop_score":         scored.get("chop_score", 0.0),
        "setup_type":         setup_type_str,
        "missing_confirmations": scored.get("missing_confirmations", []),
    }
    signal_row["ai_explanation"] = explainer.generate(signal_row, scored["breakdown"])
    new_sig_id = _write_signal(sb, signal_row)

    # ── Mark setup as promoted to CONFIRMED in lifecycle tracker ─
    try:
        _lifecycle.mark_promoted(ticker, direction, strategy_type, signal_id=None)
    except Exception:
        pass

    try:
        push.send_signal_alert(
            ticker, direction, scored["total"], "stock",
            signal_id=str(new_sig_id) if new_sig_id else None,
        )
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
            opt_sig_id = _write_option_signal(sb, opt)
            try:
                push.send_signal_alert(
                    ticker, direction, scored["total"], "option",
                    signal_id=str(opt_sig_id) if opt_sig_id else None,
                )
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
        f"[runner] {ticker} [dark_pool] score={scored.get('total', 0)}/{scored.get('threshold', 0)} "
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
        "score_breakdown":    {"detector_source": "DARK_POOL", **scored.get("breakdown", {})},
    }
    signal_row["ai_explanation"] = explainer.generate(signal_row, scored["breakdown"])
    dark_sig_id = _write_signal(sb, signal_row)

    try:
        push.send_signal_alert(
            ticker, direction, scored["total"], "stock",
            signal_id=str(dark_sig_id) if dark_sig_id else None,
        )
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
        f"[runner] {ticker} [options_flow] score={scored.get('total', 0)}/{scored.get('threshold', 0)} "
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
        "score_breakdown":    {"detector_source": "OPTIONS_FLOW", **scored.get("breakdown", {})},
    }
    signal_row["ai_explanation"] = explainer.generate(signal_row, scored["breakdown"])
    flow_sig_id = _write_signal(sb, signal_row)

    try:
        push.send_signal_alert(
            ticker, direction, scored["total"], "stock",
            signal_id=str(flow_sig_id) if flow_sig_id else None,
        )
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
        # ── Regime-first routing ─────────────────────────────────
        # RANGING or LOW_VOL regimes: try mean-reversion pipeline first.
        # If MR fires we're done; if it doesn't fire, fall through to SMC.
        regime_type = (regime or {}).get("regime_type", "")
        mr_tried = False
        if strategy_type in ("day_trade", "scalping") and regime_type in ("RANGING", "LOW_VOL"):
            fired = _process_mr_ticker(sb, ticker, config, **ctx)
            mr_tried = True
            if fired:
                return

        _process_smc_ticker(sb, ticker, config, **ctx)

        # ── Predictive setup detectors (Compression + Pullback) ────────────
        # Run AFTER SMC so they only add NEW signals on tickers SMC missed.
        # day_trade only — scalping window too short for compression patterns,
        # swing operates on a different timescale.
        if strategy_type == "day_trade":
            try:
                _process_predictive_ticker(sb, ticker, config, **ctx)
            except Exception as e:
                logger.debug(f"[runner] {ticker} predictive error: {e}")


# Predictive setups must clear this target:stop ratio. Several losers had ~1.4-
# 1.5 R:R (BKNG, NKE), which can't survive a sub-50% hit rate. (2026-05-28)
_MIN_PREDICTIVE_RR = 2.0


def _process_predictive_ticker(sb: Client, ticker: str, config: dict,
                                regime: dict = None, session: dict = None) -> None:
    """
    Parallel predictive pipeline: compression breakout + pullback continuation.
    Aims to fire at the START of moves (vs SMC which fires after confirmation).

    Uses fixed confidence_score (75) — these are structural setups, not
    multi-layer scored. Still flows through entry_gate + sl_tp_engine for
    consistent risk handling.
    """
    strategy_type = config["type"]
    regime  = regime or {}
    session = session or {}

    # Avoid duplicate fires
    if _has_active_signal(sb, ticker, strategy_type):
        return

    # Fetch bars + latest price
    try:
        from engine import alpaca_client as _alpaca
        df    = _alpaca.get_bars(ticker, timeframe=config.get("interval", "15m"), days=5)
        price = _alpaca.get_latest_price(ticker)
    except Exception:
        return
    if df is None or df.empty or not price:
        return

    # ── Stage compression zone for per-tick breakout watching ───────────
    # If this ticker is in tight consolidation (no breakout yet), register
    # its envelope so stream.on_trade can fire the INSTANT price crosses the
    # edge — instead of waiting for the next 15m scan. This is the fix for
    # missing breakouts that happen mid-bar.
    try:
        from engine import stream as _stream
        zone_staged = False
        zone = compression_detector.detect_zone(df)
        if zone is not None:
            _stream.stage_compression_zone(ticker, zone.range_high, zone.range_low, zone.atr)
            zone_staged = True
        else:
            _stream.clear_compression_zone(ticker)
        # Stage pullback reclaim level too (per-tick fire when price reclaims)
        pz = pullback_detector.detect_zone(df, current_price=price)
        if pz is not None:
            _stream.stage_pullback_zone(ticker, pz.direction, pz.reclaim_level, pz.stop_ref, pz.atr)
            zone_staged = True
        else:
            _stream.clear_pullback_zone(ticker)
        # Stage swing-high breakout levels (per-tick fire on the break)
        sz = swing_breakout_detector.detect_zone(df, current_price=price)
        if sz is not None:
            _stream.stage_swing_zone(ticker, sz.swing_high, sz.swing_low, sz.atr)
            zone_staged = True
        else:
            _stream.clear_swing_zone(ticker)
        # CRITICAL: a staged zone can ONLY fire from on_trade ticks, and Alpaca
        # pushes ticks only for SUBSCRIBED tickers. The predictive scan runs over
        # the dynamic pre-screened universe (movers), most of which are NOT in the
        # base stream subscription — so without this, zones armed on movers never
        # receive ticks and can never fire (root cause of "zones accumulate but
        # never fire", 2026-05-28). Subscribe the instant we arm a zone. Capped so
        # the live subscription can't grow without bound.
        if zone_staged:
            _ensure_stream_subscription(ticker, cap=200)
            # Record whether this ticker is currently relaxed-eligible (extended
            # past the standard cap with trend+volume confirming) so the Armed
            # Zones view can badge it. Cleared automatically when not eligible.
            try:
                _stream.set_zone_relaxed(ticker, entry_gate.momentum_relaxed_state(df, price))
            except Exception:
                pass
        else:
            try:
                _stream.set_zone_relaxed(ticker, None)
            except Exception:
                pass
        # Persist staged zones (throttled inside _persist_zones).
        _stream._persist_zones()
    except Exception:
        pass

    # Try compression breakout first
    setup_name: str  = ""
    setup_reason: str = ""
    direction:    Optional[str] = None
    try:
        comp_setup = compression_detector.detect(df, current_price=price)
    except Exception:
        comp_setup = None
    if comp_setup is not None:
        direction    = comp_setup.direction
        setup_name   = comp_setup.setup_type
        setup_reason = (f"Compression breakout — range ${comp_setup.range_low:.2f}-"
                        f"${comp_setup.range_high:.2f}, breakout +{comp_setup.breakout_pct:.2f}%")

    # Else try pullback completion
    if direction is None:
        try:
            pb_setup = pullback_detector.detect(df, current_price=price)
        except Exception:
            pb_setup = None
        if pb_setup is not None:
            direction    = pb_setup.direction
            setup_name   = pb_setup.setup_type
            setup_reason = (f"Pullback reclaim — leg {pb_setup.leg_bars}b + pullback "
                            f"{pb_setup.pullback_bars}b ({pb_setup.retracement_pct:.0%} retrace), "
                            f"reclaim ${pb_setup.swing_level:.2f}")

    if direction is None:
        return   # no predictive setup on this ticker right now

    logger.info(f"[runner] {ticker} [{strategy_type}] PREDICTIVE setup: {setup_name} {direction} @ ${price:.2f}")

    # ── Compute SL/TP using the standard engine for consistent risk ─────
    sltp = sl_tp_engine.calculate(
        direction     = direction,
        entry         = price,
        df            = df,
        regime        = regime,
        session       = session,
        gamma         = {"available": False},
        strategy_type = strategy_type,
        interval      = config.get("interval", "15m"),
    )
    if not sltp.get("valid"):
        logger.info(f"[runner] {ticker} [{strategy_type}] PREDICTIVE blocked — R:R={sltp.get('risk_reward_1', 0):.2f}")
        return
    if (sltp.get("risk_reward_1") or 0) < _MIN_PREDICTIVE_RR:
        logger.info(f"[runner] {ticker} [{strategy_type}] PREDICTIVE blocked — R:R "
                    f"{sltp.get('risk_reward_1', 0):.2f} < {_MIN_PREDICTIVE_RR}")
        return

    # ── Run the same entry gates as SMC for consistency ─────────────────
    # Compression uses the faster 5m trend gate (breakout = trend inflection);
    # pullback keeps the 15m trend gate (it's trend-following, not inflection).
    _detector = "COMPRESSION" if setup_name == "COMPRESSION_BREAKOUT" else "PULLBACK"
    entry_gate_log: dict = {}
    try:
        gate = entry_gate.check(
            ticker        = ticker, direction = direction,
            strategy_type = strategy_type,
            df_entry      = df, price = price,
            entry_tf      = config.get("interval", "15m"),
            detector      = _detector,
        )
        entry_gate_log = dict(gate.gate_log)
        if not gate.allowed:
            logger.info(f"[runner] {ticker} [{strategy_type}] PREDICTIVE blocked by gate: {' | '.join(gate.reasons)}")
            entry_gate.log_rejection(
                sb=sb, ticker=ticker, direction=direction,
                strategy_type=strategy_type, price=price,
                confidence_score=75, gate=gate,
            )
            return
    except Exception as e:
        logger.warning(f"[runner] {ticker} PREDICTIVE entry_gate error (failing open): {e}")
        entry_gate_log = {"error": str(e)}

    # ── Portfolio risk ──────────────────────────────────────────────────
    risk = risk_manager.check(sb, ticker, 75)
    if not risk["allowed"]:
        logger.info(f"[runner] {ticker} [{strategy_type}] PREDICTIVE blocked — portfolio: {risk['block_reason']}")
        return

    _tape_b = _tape_bonus(ticker)
    adjusted_score = max(0, min(100, 75 + _tape_b.get("bonus", 0)))

    signal_row = {
        "ticker":             ticker,
        "direction":          direction,
        "entry_price":        round(price, 2),
        "stop_loss":          sltp["stop_loss"],
        "target_one":         sltp["target_one"],
        "target_two":         sltp["target_two"],
        "confidence_score":   adjusted_score,
        "confidence_factors": [setup_reason] + _tape_b.get("reasons", []),
        "timeframe":          config["interval"],
        "strategy_type":      strategy_type,
        "status":             "active",
        "ai_explanation":     setup_reason,
        "regime_type":        regime.get("regime_type", ""),
        "session_mode":       session.get("mode", ""),
        "confidence_tier":    risk["confidence_tier"],
        "position_multiplier": risk["position_mult"],
        "sl_adjustments":     sltp.get("adjustments", []),
        "risk_reward":        sltp["risk_reward_1"],
        "score_breakdown":    {
            "detector_source": "COMPRESSION" if setup_name == "COMPRESSION_BREAKOUT" else "PULLBACK",
            "predictive_setup": setup_name,
            "predictive_reason": setup_reason,
            "entry_gate":      entry_gate_log,
            "sl_width_pct":    round(abs(price - sltp["stop_loss"]) / price * 100, 3),
            "atr_used":        round(sltp.get("atr", 0), 4),
            "adr_used":        round(sltp.get("adr", 0), 4),
            "tape_bonus":      _tape_b,
        },
        "confidence_grade":   "B+",   # predictive setups default tier
        "setup_type":         setup_name,
    }
    new_sig_id = _write_signal(sb, signal_row)
    try:
        push.send_signal_alert(
            ticker, direction, adjusted_score, "stock",
            signal_id=str(new_sig_id) if new_sig_id else None,
        )
    except Exception:
        pass


def _fire_per_tick_predictive(ticker: str, direction: str, price: float,
                              detector: str, setup_type: str, label: str,
                              armed_ts: float | None = None,
                              breakout_level: float | None = None) -> None:
    """
    Shared fire path for predictive detectors (compression / pullback / swing).
    Called from stream.on_bar when a 1-minute bar CLOSES beyond a staged level
    (body confirmation), instead of waiting for the next 15m scan.

    `armed_ts`: epoch seconds when the zone was staged — stored as armed_at so
    the chart can mark the accumulation point alongside the fire point.

    Self-contained: own Supabase client, fetches bars for SL/TP, runs the same
    entry_gate + risk checks as scan-fired signals. Deduped by active-signal
    check + DB unique index.
    """
    try:
        sb = create_client(os.environ["SUPABASE_URL"], _supabase_key())
    except Exception as e:
        logger.debug(f"[runner] {detector} fire — supabase init failed: {e}")
        return

    strategy_type = "day_trade"
    if _has_active_signal(sb, ticker, strategy_type):
        return

    try:
        from engine import alpaca_client as _alpaca
        df = _alpaca.get_bars(ticker, timeframe="15m", days=5)
    except Exception:
        return
    if df is None or df.empty:
        return

    setup_reason = f"{label} (per-tick) — {direction} @ ${price:.2f}"
    logger.info(f"[runner] {ticker} {label.upper()} (per-tick) {direction} @ ${price:.2f}")

    sltp = sl_tp_engine.calculate(
        direction=direction, entry=price, df=df,
        regime={}, session={}, gamma={"available": False},
        strategy_type=strategy_type, interval="15m",
    )
    if not sltp.get("valid"):
        logger.info(f"[runner] {ticker} {detector} — R:R too low, skipping")
        return

    # Retest entry: place the stop just BELOW the broken level (LONG) / ABOVE it
    # (SHORT) with an ATR buffer, instead of a tight % off the entry. Entry sits
    # near the level on the retest, so this gives a sane, level-based stop that a
    # normal pullback won't trip (fixes the BKNG 0.2%-stop fade, 2026-05-28).
    if breakout_level:
        _atr = float(sltp.get("atr", 0) or 0)
        buf  = _atr * 0.5 if _atr > 0 else price * 0.003
        if direction == "LONG":
            sltp["stop_loss"] = round(breakout_level - buf, 2)
        else:
            sltp["stop_loss"] = round(breakout_level + buf, 2)
        risk = abs(price - sltp["stop_loss"])
        if risk > 0:
            sltp["risk_reward_1"] = round(abs(sltp["target_one"] - price) / risk, 2)

    if (sltp.get("risk_reward_1") or 0) < _MIN_PREDICTIVE_RR:
        logger.info(f"[runner] {ticker} {detector} — R:R {sltp.get('risk_reward_1', 0):.2f} "
                    f"< {_MIN_PREDICTIVE_RR}, skipping")
        return

    entry_gate_log: dict = {}
    try:
        gate = entry_gate.check(
            ticker=ticker, direction=direction, strategy_type=strategy_type,
            df_entry=df, price=price, entry_tf="15m", detector=detector,
        )
        entry_gate_log = dict(gate.gate_log)
        if not gate.allowed:
            logger.info(f"[runner] {ticker} {detector} blocked by gate: {' | '.join(gate.reasons)}")
            entry_gate.log_rejection(
                sb=sb, ticker=ticker, direction=direction, strategy_type=strategy_type,
                price=price, confidence_score=75, gate=gate,
            )
            return
    except Exception as e:
        logger.warning(f"[runner] {ticker} {detector} gate error (failing open): {e}")
        entry_gate_log = {"error": str(e)}

    risk = risk_manager.check(sb, ticker, 75)
    if not risk["allowed"]:
        logger.info(f"[runner] {ticker} {detector} blocked — portfolio: {risk['block_reason']}")
        return

    _tape_b = _tape_bonus(ticker)
    adjusted_score = max(0, min(100, 75 + _tape_b.get("bonus", 0)))

    signal_row = {
        "ticker":             ticker,
        "direction":          direction,
        "entry_price":        round(price, 2),
        "stop_loss":          sltp["stop_loss"],
        "target_one":         sltp["target_one"],
        "target_two":         sltp["target_two"],
        "confidence_score":   adjusted_score,
        "confidence_factors": [setup_reason] + _tape_b.get("reasons", []),
        "timeframe":          "15m",
        "strategy_type":      strategy_type,
        "status":             "active",
        "ai_explanation":     setup_reason,
        "regime_type":        "",
        "session_mode":       "",
        "confidence_tier":    risk["confidence_tier"],
        "position_multiplier": risk["position_mult"],
        "sl_adjustments":     sltp.get("adjustments", []),
        "risk_reward":        sltp["risk_reward_1"],
        "score_breakdown":    {
            "detector_source":   detector,
            "predictive_setup":  setup_type,
            "predictive_reason": setup_reason,
            "fire_path":         "bar_close_1m",
            "armed_at":          (datetime.fromtimestamp(armed_ts, tz=timezone.utc).isoformat()
                                  if armed_ts else None),
            "entry_gate":        entry_gate_log,
            "sl_width_pct":      round(abs(price - sltp["stop_loss"]) / price * 100, 3),
            "atr_used":          round(sltp.get("atr", 0), 4),
            "tape_bonus":        _tape_b,
        },
        "confidence_grade":   "B+",
        "setup_type":         setup_type,
    }
    new_sig_id = _write_signal(sb, signal_row)
    try:
        push.send_signal_alert(ticker, direction, adjusted_score, "stock",
                               signal_id=str(new_sig_id) if new_sig_id else None)
    except Exception:
        pass


def fire_compression_breakout(ticker: str, direction: str, price: float,
                              armed_ts: float | None = None, level: float | None = None) -> None:
    """Compression breakout fire on retest (called from stream.on_bar)."""
    _fire_per_tick_predictive(ticker, direction, price,
                              detector="COMPRESSION",
                              setup_type="COMPRESSION_BREAKOUT",
                              label="Compression breakout", armed_ts=armed_ts,
                              breakout_level=level)


def fire_pullback_reclaim(ticker: str, direction: str, price: float,
                          armed_ts: float | None = None, level: float | None = None) -> None:
    """1m-close pullback reclaim fire (called from stream.on_bar)."""
    _fire_per_tick_predictive(ticker, direction, price,
                              detector="PULLBACK",
                              setup_type="PULLBACK_CONTINUATION",
                              label="Pullback reclaim", armed_ts=armed_ts,
                              breakout_level=level)


def fire_swing_breakout(ticker: str, direction: str, price: float,
                        armed_ts: float | None = None, level: float | None = None) -> None:
    """Swing-high breakout fire on retest (called from stream.on_bar)."""
    _fire_per_tick_predictive(ticker, direction, price,
                              detector="SWING_BREAKOUT",
                              setup_type="SWING_BREAKOUT",
                              label="Swing-high breakout", armed_ts=armed_ts,
                              breakout_level=level)


# ---------------------------------------------------------------------------
# Auto-close logic
# ---------------------------------------------------------------------------

def _close_signals(sb: Client) -> None:
    """
    Close signals that hit target/stop or exceeded their max hold time.
    Scalping signals expire after 30 min; swing trade after 10 days.

    Uses Alpaca batch price fetch (one API call for all active tickers) then
    yfinance per-ticker as fallback — never creates a new Alpaca client per call.
    """
    from engine import alpaca_client as _alpaca

    now = datetime.now(timezone.utc)

    # ── Stock signals ──
    try:
        rows = sb.table("signals").select("*").eq("status", "active").execute().data
    except Exception as e:
        logger.error(f"[closer] fetch signals failed: {e}")
        rows = []

    # ── Batch price fetch: one Alpaca call for all active tickers ─────────────
    non_expired_tickers = []
    for sig in rows:
        created   = datetime.fromisoformat(sig["created_at"].replace("Z", "+00:00"))
        strategy  = sig.get("strategy_type") or "day_trade"
        hold_hours = STRATEGY_MAX_HOLD_HOURS.get(strategy, 48.0)
        if created >= now - timedelta(hours=hold_hours):
            non_expired_tickers.append(sig["ticker"])

    price_map: dict[str, float] = {}
    if non_expired_tickers:
        # Deduplicate
        unique_tickers = list(dict.fromkeys(non_expired_tickers))
        price_map = _alpaca.get_latest_prices(unique_tickers)
        # yfinance fallback for any ticker Alpaca missed
        missing = [t for t in unique_tickers if t not in price_map]
        if missing:
            try:
                import yfinance as yf
                for t in missing:
                    p = yf.Ticker(t).fast_info.last_price
                    if p:
                        price_map[t] = float(p)
            except Exception:
                pass

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
            close_price = price_map.get(sig["ticker"])
            if close_price:
                if sig["direction"] == "LONG":
                    if close_price >= sig["target_two"]:  reason = "target_hit"
                    elif close_price <= sig["stop_loss"]: reason = "stop_hit"
                else:
                    if close_price <= sig["target_two"]:  reason = "target_hit"
                    elif close_price >= sig["stop_loss"]: reason = "stop_hit"

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

    # Batch fetch prices for all non-expired option signal underlyings
    opt_non_expired = [
        sig["ticker"]
        for sig in opt_rows
        if datetime.fromisoformat(sig["created_at"].replace("Z", "+00:00")) >= now - timedelta(hours=24)
    ]
    opt_price_map: dict[str, float] = {}
    if opt_non_expired:
        unique_opt = list(dict.fromkeys(opt_non_expired))
        opt_price_map = _alpaca.get_latest_prices(unique_opt)
        missing_opt = [t for t in unique_opt if t not in opt_price_map]
        if missing_opt:
            try:
                import yfinance as yf
                for t in missing_opt:
                    p = yf.Ticker(t).fast_info.last_price
                    if p:
                        opt_price_map[t] = float(p)
            except Exception:
                pass

    option_cutoff = now - timedelta(hours=24)
    for sig in opt_rows:
        created = datetime.fromisoformat(sig["created_at"].replace("Z", "+00:00"))
        reason  = None

        if created < option_cutoff:
            reason = "expired"
        else:
            try:
                price      = opt_price_map.get(sig["ticker"])
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

    # ── Market-closed short-circuit ───────────────────────────────────────────
    # Skip the entire scan on weekends / NYSE holidays / pre-market /
    # after-hours. The session classifier would block every signal anyway,
    # but doing it here avoids the expensive pre-screener + Alpaca calls.
    from engine.session_classifier import is_market_open_now, is_market_open_today
    if not is_market_open_today():
        logger.info(f"[runner] {strategy_type} scan skipped — market closed today (holiday/weekend)")
        return
    if not is_market_open_now():
        logger.info(f"[runner] {strategy_type} scan skipped — outside trading hours")
        return

    # ── Dynamic ticker list via pre-screener ──────────────────────────────────
    # Run the fast Alpaca snapshot screen to find the ~50 tickers showing
    # real momentum or volume activity right now.  Full SMC only runs on those.
    # Scalping uses a tighter list — needs the highest-liquidity names only.
    if strategy_type == "scalping":
        tickers = SCALP_TICKERS
    else:
        try:
            tickers = prescreener.screen(max_results=150)
        except Exception as e:
            logger.warning(f"[runner] Pre-screener failed: {e} — using core tickers")
            tickers = prescreener.CORE_TICKERS

        # Apply pre-market priority ordering (no-op outside 9:30–10:30 AM ET window
        # or when cache is empty).  High-watch-score tickers bubble to the front
        # so SMC gets the most interesting setups first each morning.
        try:
            tickers = premarket_scanner.get_priority_tickers(tickers)
        except Exception as _pm_err:
            logger.debug(f"[runner] Pre-market prioritisation skipped: {_pm_err}")

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
    """Track open signal results, auto-close hits, expire stale setups. Runs every 15 min."""
    # Skip on closed-market days. Prices aren't moving, nothing can hit a
    # level, and the signal_monitor inside this function would just churn
    # API calls for no information gain.
    from engine.session_classifier import is_market_open_today
    if not is_market_open_today():
        logger.info("[runner] Maintenance skipped — market closed today")
        return
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
    # ── Expire stale WATCHLIST / DEVELOPING setups ────────────
    try:
        expired = _lifecycle.expire_stale_setups()
        if expired:
            logger.info(f"[runner] Lifecycle: expired {expired} stale setups")
    except Exception as e:
        logger.debug(f"[runner] lifecycle expire skipped: {e}")
    logger.info("[runner] Maintenance complete")


def _run_analytics_report() -> None:
    """
    Generate daily analytics report (5:30 PM ET, after market close).
    Computes win rate, R-multiples, expectancy, false-positive stats.
    Stores results back to Supabase for the Analytics tab.
    """
    logger.info("[runner] ═══ Daily analytics report started ═══")
    try:
        sb = _supabase()
        report = signal_analytics.generate_report(sb, days=30)
        quality_flags = report.get("quality_flags", [])
        critical = [f for f in quality_flags if f.startswith("CRITICAL")]
        warnings = [f for f in quality_flags if f.startswith("WARNING")]
        logger.info(
            f"[runner] Analytics: win_rate={report['overall'].get('win_rate', 0):.1%} "
            f"expectancy={report['overall'].get('expectancy', 0):.2f}R "
            f"signals={report['overall'].get('total_signals', 0)} "
            f"({len(critical)} CRITICAL, {len(warnings)} WARNINGS)"
        )
        if critical:
            for flag in critical:
                logger.warning(f"[runner] Analytics flag: {flag}")
        # Persist summary to analytics_reports table (graceful skip if table missing)
        try:
            sb.table("analytics_reports").insert({
                "report_date":    datetime.now(timezone.utc).date().isoformat(),
                "overall":        report["overall"],
                "by_strategy":    report.get("by_strategy", {}),
                "quality_flags":  quality_flags,
            }).execute()
        except Exception as _db_err:
            logger.debug(f"[runner] analytics_reports insert skipped: {_db_err}")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[runner] Analytics report failed: {e}", exc_info=True)
    logger.info("[runner] ═══ Daily analytics report complete ═══")


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


def _run_gate_validator() -> None:
    """
    Nightly entry-gate rejection validator. Walks unjudged rows in
    entry_gate_rejections and backfills would_have_won + realized_pnl_pct
    via historical bar simulation. Runs daily at 3 AM UTC.
    """
    logger.info("[runner] ═══ Entry-gate rejection validator started ═══")
    try:
        from engine import gate_validator
        sb = create_client(os.environ["SUPABASE_URL"], _supabase_key())
        result = gate_validator.validate_batch(sb, limit=500)
        logger.info(f"[runner] ═══ Validator done — {result} ═══")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[runner] Gate validator failed: {e}", exc_info=True)


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

    # ── Market-hours signal monitor: every 5 min during full trading day ────
    # Runs signal_monitor only (no strategy scans). Fires throughout the trading
    # day so status tracking, T1 breakeven moves, and intelligent early booking
    # react within 5 min instead of waiting for the coarse 15-min maintenance
    # cycle. The 15-min maintenance still runs track_signals + _close_signals in
    # addition to signal_monitor.
    #
    # Market hours window: 9:30 AM (570 min) to 4:05 PM (965 min) ET, Mon-Fri.
    # The 35-min cushion past 4:00 PM ensures the 3:30 PM force-close job
    # fires even if the scheduler fires slightly late.
    def _market_monitor_job():
        from zoneinfo import ZoneInfo as _ZI
        from datetime import datetime as _dt
        _now = _dt.now(_ZI("America/New_York"))
        _mins = _now.hour * 60 + _now.minute
        # 9:30 AM = 570 min, 4:05 PM = 965 min
        if 570 <= _mins <= 965 and _now.weekday() < 5:
            try:
                signal_monitor.run()
            except Exception as _e:
                logger.error(f"[runner] Market monitor failed: {_e}")

    scheduler.add_job(
        _market_monitor_job,
        trigger=IntervalTrigger(minutes=5),
        id="eod_monitor",
        name="Market-hours signal monitor (5-min, 9:30 AM–4:05 PM ET)",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled market-hours signal monitor every 5 min (9:30 AM–4:05 PM ET)")

    # ── Day trade scan every 10 min during market hours ─────────────────────
    # Stream.py fires scans at 15-min bar-close events but a stock that starts
    # moving at minute 7 won't be picked up until minute 15. Running a full
    # day_trade scan every 10 min with a freshly-screened ticker list catches
    # new movers up to 5 min sooner. The 15m bar data is unchanged between closes
    # but the prescreener will surface newly-active tickers for SMC analysis.
    day_trade_config = next(c for c in STRATEGY_CONFIGS if c["type"] == "day_trade")

    def _intraday_scan_job():
        from zoneinfo import ZoneInfo as _ZI
        from datetime import datetime as _dt
        _now = _dt.now(_ZI("America/New_York"))
        _mins = _now.hour * 60 + _now.minute
        # 9:30 AM = 570, 3:55 PM = 955 — stop 5 min before EOD force-close
        if 570 <= _mins <= 955 and _now.weekday() < 5:
            try:
                _run_strategy_scan(day_trade_config)
            except Exception as _e:
                logger.error(f"[runner] 10-min day_trade scan failed: {_e}")

    scheduler.add_job(
        _intraday_scan_job,
        trigger=IntervalTrigger(minutes=10),
        id="day_trade_10min",
        name="Day trade scan (10-min, 9:30 AM–3:55 PM ET)",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled day_trade scan every 10 min (9:30 AM–3:55 PM ET)")

    # ── Pre-market scans — 8:00 AM ET (12:00 UTC) and 9:00 AM ET (13:00 UTC) ──
    from apscheduler.triggers.cron import CronTrigger
    scheduler.add_job(
        premarket_scanner.run_8am_scan,
        trigger=CronTrigger(
            day_of_week="mon-fri", hour=12, minute=0, timezone="UTC"
        ),
        id="premarket_8am",
        name="Pre-market scan 8:00 AM ET",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled pre-market scan at 8:00 AM ET (Mon-Fri)")

    scheduler.add_job(
        premarket_scanner.run_9am_scan,
        trigger=CronTrigger(
            day_of_week="mon-fri", hour=13, minute=0, timezone="UTC"
        ),
        id="premarket_9am",
        name="Pre-market scan 9:00 AM ET",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled pre-market scan at 9:00 AM ET (Mon-Fri)")

    # ── Daily analytics report — 5:30 PM ET (21:30 UTC) Mon-Fri ─────────
    scheduler.add_job(
        _run_analytics_report,
        trigger=CronTrigger(
            day_of_week="mon-fri", hour=21, minute=30, timezone="UTC"
        ),
        id="analytics_report",
        name="SignalBolt daily analytics report",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled daily analytics report (5:30 PM ET, Mon-Fri)")

    # ── Weekly self-learning optimization (Sunday 2 AM UTC) ──────────────
    scheduler.add_job(
        _run_weight_optimization,
        trigger=CronTrigger(day_of_week="sun", hour=2, minute=0, timezone="UTC"),
        id="weight_optimization",
        name="SignalBolt weight optimizer",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled weekly weight optimization (Sunday 2:00 AM UTC)")

    # ── Pre-close entry-gate rejection validator (19:30 UTC = 2:30 PM CDT) ─
    # 30 min before market close so admin has time to review results and
    # decide any tuning for tomorrow while the day is still fresh. Trade-offs:
    # rejections from the last 30 min won't have hold windows elapsed yet
    # (skipped, re-judged tomorrow); morning day_trade signals that closed
    # via SL/TP intraday are fully judged.
    scheduler.add_job(
        _run_gate_validator,
        trigger=CronTrigger(hour=19, minute=30, timezone="UTC"),
        id="gate_validator",
        name="SignalBolt entry-gate rejection validator",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled entry-gate validator (19:30 UTC / 2:30 PM CDT)")

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
