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
from engine.tracker import track_signals, result_from_pnl_pct
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
    "breakdown":      240.0,  # 10 days — bearish swing short/put from a breakdown
    "breakout":       240.0,  # 10 days — bullish swing long/call from a breakout
    "turnaround":     240.0,  # 10 days — bullish swing long/call from a cycle bottom
    "peak":           240.0,  # 10 days — bearish swing short/put from a cycle top
    # Predictive "forming" detectors are swing entries that fire EARLIER than the
    # confirmed move — they target the same multi-day move, so they inherit the
    # parent's 10-day swing window (without this they fell back to the 48h default
    # and were time-expired before the pattern could resolve). 0.25x size; real
    # exits are still thesis-driven (stop / target / trail / structure-reversal).
    "breakdown_forming": 240.0,
    "distrib_forming":   240.0,
    "peak_forming":      240.0,
    "turn_forming":      240.0,
    "accum_forming":     240.0,
    "earnings":       48.0,   # 2 days — pre/post earnings move
    "short_squeeze":  24.0,   # 1 day — squeeze resolves quickly
    "position_trade": 720.0,  # 30 days — macro position
    "deep_value":     8760.0, # 1 year — crash/deep-value long-term hold (also fired
                              # management_mode='manual', so the expiry skips it anyway)
    "options_flow":   8.0,
    "dark_pool":      8.0,
}


def is_past_max_hold(created: datetime, strategy: str) -> bool:
    """
    True if a signal has exceeded its strategy's max-hold backstop.

    The time cap is only a backstop — real exits are thesis-driven (stop /
    target / trailing stop / structure-reversal). For MULTI-DAY holds (>=24h
    window, e.g. swing 240h, position 720h) we count TRADING days so weekends
    and holidays don't consume the window, and we NEVER report expired on a
    non-trading day — so nothing closes over a Saturday/Sunday/holiday. Intraday
    strategies (same-session) keep the simple wall-clock window.
    """
    hold_hours = STRATEGY_MAX_HOLD_HOURS.get(strategy, 48.0)
    now = datetime.now(timezone.utc)
    if hold_hours >= 24:
        from engine import session_classifier
        if not session_classifier.is_market_open_today():
            return False   # never expire on a weekend / holiday
        elapsed_td = session_classifier.trading_days_between(created, now)
        return elapsed_td >= (hold_hours / 24.0)
    return (now - created) > timedelta(hours=hold_hours)


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
            # No running loop in THIS thread (the sync APScheduler scan thread).
            # Schedule the subscribe onto the WORKER's stream loop so the ticker
            # is ACTUALLY subscribed on Alpaca NOW — not merely queued to
            # _pending_tickers, which only drains on the next reconnect. A fired
            # ticker used to sit unsubscribed for hours while the stream stayed
            # up → frozen price + dark RT stop/target checks (HOOD 2026-06-04).
            sl = getattr(_stream, "_stream_loop", None)
            if sl is not None and sl.is_running():
                asyncio.run_coroutine_threadsafe(
                    _stream.subscribe_extra_tickers([ticker]), sl
                )
            else:
                # Stream not up yet — queue for the initial connect to apply.
                _stream._subscribed_tickers.add(ticker)
                _stream._pending_tickers.add(ticker)
    except Exception as e:
        logger.debug(f"[runner] stream subscribe failed for {ticker}: {e}")


_MIN_SIGNAL_PRICE = 2.0   # hard floor — penny/sub-penny names have wide spreads


def _is_untradeable(ticker: str, price, strategy_type: str | None = None) -> bool:
    """Block penny stocks, warrants/units/rights, AND the dangerous leveraged/inverse
    products from EVER firing a signal. Belt-and-suspenders behind the prescreener —
    covers every fire path (SMC, breakout/breakdown, momentum, predictive, cycle).
    HUBCW (a $0.05 warrant) lost -31.8% in 9 min.

    Two-tier leveraged policy (see engine/leveraged_etfs.py):
      • ALWAYS blocked — single-stock 2x/3x (TSLL/NVDL…), vol ETNs (UVXY/VXX…),
        commodity/metal futures (BOIL/NUGT…), leveraged bonds/EM. Uniquely path-
        dependent; no clean broad-index to track.
      • Leveraged BROAD-INDEX / US-sector equity (TQQQ/SQQQ/SOXL/SPXL…) — ALLOWED
        for short-horizon signals, but BLOCKED on months-horizon strategies
        (deep_value/position_trade) where 3x decay ruins a multi-month hold.
    Passing strategy_type enables the long-horizon block; omit it for the
    strategy-agnostic always-blocked check."""
    sym = (ticker or "").upper()
    if len(sym) == 5 and sym[-1] in ("W", "U", "R"):   # warrant / unit / rights
        return True
    from engine.leveraged_etfs import should_block_signal
    if should_block_signal(sym, strategy_type):        # always-blocked + lev-index×long-horizon
        return True
    try:
        if price is not None and float(price) < _MIN_SIGNAL_PRICE:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _live_regime_type() -> str:
    """Current regime label to stamp on a signal at fire when the caller didn't
    pass one — so every detector signal is regime-sliceable. Never empty/raises."""
    try:
        from engine import signal_telemetry
        return signal_telemetry.live_regime_type()
    except Exception:
        return "RANGING"


def _write_signal(sb: Client, row: dict) -> str | None:
    """Insert signal row, log the 'fired' event, and return the new signal ID."""
    if _is_untradeable(row.get("ticker", ""), row.get("entry_price"), row.get("strategy_type")):
        logger.info(
            f"[runner] BLOCKED untradeable signal {row.get('ticker')} "
            f"@ {row.get('entry_price')} [{row.get('strategy_type','?')}] "
            f"(penny / warrant / always-blocked leveraged / leveraged-index on long-horizon) — not firing"
        )
        return None
    # Capture the ORIGINAL stop at fire time. The monitors mutate stop_loss in
    # place as they trail it, so without this the fired level is lost — and the
    # UI can't show users that the stop was RAISED. Done here so every fire path
    # (SMC, momentum, predictive, options…) is covered in one place.
    try:
        if isinstance(row.get("score_breakdown"), dict) and row.get("stop_loss") is not None:
            row["score_breakdown"].setdefault("initial_stop", row["stop_loss"])
    except Exception:
        pass
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
    # Options on leveraged ETFs are leverage-on-leverage. Always-blocked products
    # never fire; leveraged-index underlyings are blocked on the months-horizon
    # LEAP/position strategies (3x decay over a multi-month option hold is fatal).
    if _is_untradeable(row.get("ticker", ""), row.get("underlying_price"), row.get("strategy_type")):
        logger.info(f"[runner] BLOCKED option signal {row.get('ticker')} "
                    f"[{row.get('strategy_type','?')}] "
                    f"(penny / warrant / always-blocked leveraged / leveraged-index on long-horizon) — not firing")
        return None
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
            "regime_type":         (regime.get("regime_type") or _live_regime_type()),
            "session_mode":        session.get("mode", ""),
            "confidence_tier":     risk["confidence_tier"],
            "position_multiplier": risk["position_mult"],
            "setup_type":          "VWAP_MEAN_REVERSION",
            "confidence_grade":    "B+" if mr.score >= 74 else "B",
            "chop_score":          0.0,   # MR signals trade IN chop — no chop penalty
            "score_breakdown":     {"detector_source": "MEAN_REVERSION",
                                    "mr_score": mr.score, "mr_passes": list(mr.passes)},
        })
        explainer.attach_narrative(signal_row, signal_row["score_breakdown"])
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

    # Late-session / after-hours guard for intraday strategies. Event-driven
    # bar closes keep triggering scans after 16:00 ET (Alpaca EXTO bars), which
    # produced after-hours LONG chases (e.g. CRWD @674 at 4:31–5:20 PM ET). A
    # day_trade entered that late has no session left. Swing/position exempt.
    if strategy_type in _INTRADAY_STRATEGIES and not _intraday_entry_window_open():
        logger.debug(f"[runner] {ticker} [{strategy_type}] — outside intraday entry window, skipping")
        return

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

    # ── Market-regime alignment (was predictive-only; now SMC too) ───────
    # SMC had NO regime gate — which is how SIDU fired a counter-trend SHORT
    # into a TRENDING_BULL and lost the full move (2026-05-28). Block LONG in
    # bear/risk-off/panic regimes and SHORT in a strong bull. Logged as a
    # rejection (detector=SMC) so the scorecard can later confirm the filter
    # is correctly skipping losers vs killing winners.
    _rt = regime.get("regime_type", "RANGING")
    _regime_blocks = (
        (direction == "LONG"  and _rt in ("TRENDING_BEAR", "RISK_OFF", "PANIC")) or
        (direction == "SHORT" and _rt == "TRENDING_BULL")
    )
    if _regime_blocks:
        reason = f"market regime {_rt} against {direction}"
        logger.info(f"[runner] {ticker} [{strategy_type}] BLOCKED — {reason}")
        try:
            _gr = entry_gate.GateResult(allowed=False, reasons=[reason],
                                        gate_log={"market_regime": f"fail: {reason}"})
            entry_gate.log_rejection(sb=sb, ticker=ticker, direction=direction,
                                     strategy_type=strategy_type, price=price,
                                     confidence_score=0, gate=_gr, detector="SMC")
        except Exception:
            pass
        return

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
            has_catalyst  = has_news,   # breaking news → wider overextension cap
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
        "regime_type":        (regime.get("regime_type") or _live_regime_type()),
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
    # Supply/demand zone bounds for the chart's SUPPORT/RESISTANCE box.
    _zone = smc.zone_bounds(analysis)
    if _zone:
        signal_row["score_breakdown"]["zone"] = _zone
    explainer.attach_narrative(signal_row, scored["breakdown"])
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
        "regime_type":        (regime.get("regime_type") or _live_regime_type()),
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
    explainer.attach_narrative(signal_row, scored["breakdown"])
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
        "regime_type":        (regime.get("regime_type") or _live_regime_type()),
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
    explainer.attach_narrative(signal_row, scored["breakdown"])
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

# Per-tick predictive fires are day_trade (close same session). Don't open a
# new entry in the last stretch of the session — there's no time left for the
# move to work and EOD close sweeps it flat. The MSTR COMP SHORT that fired at
# 4:01 PM ET (1 min AFTER the close) and closed at 4:03 PM flat exposed the
# missing guard. RTH is 9:30–16:00 ET; block new fires from 15:45 ET on.
_PREDICTIVE_ENTRY_OPEN_MIN  = 9 * 60 + 30   # 09:30 ET
_PREDICTIVE_ENTRY_CLOSE_MIN = 15 * 60 + 45  # 15:45 ET


# Intraday strategies that must not open new entries late / after hours.
# Swing / position trades hold for days, so a late entry is fine for them.
_INTRADAY_STRATEGIES = {"day_trade", "scalping", "options_flow", "dark_pool"}


def _intraday_entry_window_open() -> bool:
    """True only during the intraday entry window (09:30–15:45 ET, weekday).
    Used to block day_trade/scalping entries late in or after the session —
    event-driven bar closes still fire after 16:00 ET (e.g. CRWD after-hours
    chases) so the schedule alone isn't enough."""
    try:
        from zoneinfo import ZoneInfo as _ZI
        now = datetime.now(_ZI("America/New_York"))
        if now.weekday() > 4:                       # Sat/Sun
            return False
        mins = now.hour * 60 + now.minute
        return _PREDICTIVE_ENTRY_OPEN_MIN <= mins < _PREDICTIVE_ENTRY_CLOSE_MIN
    except Exception:
        return True   # fail open — never worse than today's behavior


# Backwards-compatible alias for the predictive fire path.
def _predictive_entry_window_open() -> bool:
    return _intraday_entry_window_open()


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
        from engine import zone_history as _zh
        zone_staged = False
        # Track fresh arms so we log each arming exactly once to history.
        _armed_new: list[dict] = []
        # Relaxed-eligible state (computed once, reused for the history rows).
        try:
            _relaxed = entry_gate.momentum_relaxed_state(df, price) or {}
        except Exception:
            _relaxed = {}
        _is_relaxed = bool(_relaxed.get("eligible"))
        _ext_atr    = _relaxed.get("ext_atr")

        zone = compression_detector.detect_zone(df)
        if zone is not None:
            if _stream.stage_compression_zone(ticker, zone.range_high, zone.range_low, zone.atr):
                _armed_new.append({"detector": "COMPRESSION", "direction": None,
                                   "armed_level": None,
                                   "range_high": zone.range_high, "range_low": zone.range_low,
                                   "atr": zone.atr})
            zone_staged = True
        else:
            _stream.clear_compression_zone(ticker)
        # Stage pullback reclaim level too (per-tick fire when price reclaims)
        pz = pullback_detector.detect_zone(df, current_price=price)
        if pz is not None:
            if _stream.stage_pullback_zone(ticker, pz.direction, pz.reclaim_level, pz.stop_ref, pz.atr):
                _armed_new.append({"detector": "PULLBACK", "direction": pz.direction,
                                   "armed_level": pz.reclaim_level,
                                   "range_high": None, "range_low": None, "atr": pz.atr})
            zone_staged = True
        else:
            _stream.clear_pullback_zone(ticker)
        # Stage swing-high breakout levels (per-tick fire on the break)
        sz = swing_breakout_detector.detect_zone(df, current_price=price)
        if sz is not None:
            if _stream.stage_swing_zone(ticker, sz.swing_high, sz.swing_low, sz.atr):
                _armed_new.append({"detector": "SWING_BREAKOUT", "direction": None,
                                   "armed_level": sz.swing_high, "range_high": sz.swing_high,
                                   "range_low": sz.swing_low, "atr": sz.atr})
            zone_staged = True
        else:
            _stream.clear_swing_zone(ticker)

        # Log fresh arms to lifecycle history (best-effort, off the hot path).
        for _a in _armed_new:
            _zh.log_armed(sb, ticker=ticker, detector=_a["detector"],
                          direction=_a["direction"], armed_level=_a["armed_level"],
                          range_high=_a["range_high"], range_low=_a["range_low"],
                          atr=_a["atr"], relaxed=_is_relaxed, ext_atr=_ext_atr)
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
                _stream.set_zone_relaxed(ticker, _relaxed or None)
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

    # ── EMA reclaim / trend-day continuation ─────────────────────────────
    # Catches the high-momentum first-pullback-to-9EMA entry (HOOD/CRWD trend
    # days) that SMC + generic gates were rejecting. Fires on the bar that
    # completes the pullback-hold; rides on a 20-EMA trail (signal_monitor).
    try:
        from engine import ema_reclaim_detector
        er = ema_reclaim_detector.detect(df, price)
        if er is not None:
            logger.info(f"[runner] {ticker} EMA_RECLAIM {er.direction} @ ${price:.2f} "
                        f"(ema9={er.ema9:.2f} ema20={er.ema20:.2f} rsi={er.rsi:.0f})")
            fire_ema_reclaim(ticker, er.direction, price, level=er.stop_ref)
    except Exception as e:
        logger.debug(f"[runner] {ticker} EMA_RECLAIM error: {e}")

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
                confidence_score=75, gate=gate, detector=detector,
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
        "regime_type":        (regime.get("regime_type") or _live_regime_type()),
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

    # Late-session / RTH guard — a day_trade entry opened in the last 15 min
    # (or after the close on a late-delivered bar) has no time to work and gets
    # swept flat by EOD close. Block new fires outside 09:30–15:45 ET.
    if not _predictive_entry_window_open():
        logger.info(f"[runner] {ticker} {detector} — outside entry window (>=15:45 ET), skipping")
        return

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
            has_catalyst=_has_recent_news(ticker),   # breaking news → wider overextension cap
        )
        entry_gate_log = dict(gate.gate_log)
        if not gate.allowed:
            logger.info(f"[runner] {ticker} {detector} blocked by gate: {' | '.join(gate.reasons)}")
            entry_gate.log_rejection(
                sb=sb, ticker=ticker, direction=direction, strategy_type=strategy_type,
                price=price, confidence_score=75, gate=gate, detector=detector,
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
        "regime_type":        _live_regime_type(),
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
        from engine import zone_history as _zh
        _zh.mark_fired(sb, ticker=ticker, detector=detector, fired_signal_id=new_sig_id)
    except Exception:
        pass
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


def fire_ema_reclaim(ticker: str, direction: str, price: float,
                     armed_ts: float | None = None, level: float | None = None) -> None:
    """EMA reclaim / trend-continuation fire (first pullback to 9 EMA holds).
    Rides on a 20-EMA trail — see signal_monitor 4b. `level` = pullback extreme
    for the level-based stop."""
    _fire_per_tick_predictive(ticker, direction, price,
                              detector="EMA_RECLAIM",
                              setup_type="EMA_RECLAIM",
                              label="EMA reclaim", armed_ts=armed_ts,
                              breakout_level=level)


# ---------------------------------------------------------------------------
# Systematic momentum / trend-following (cross-sectional, daily bars)
# ---------------------------------------------------------------------------
_MOMENTUM_MAX_LONGS  = 8
_MOMENTUM_MAX_SHORTS = 4
_MOMENTUM_ATR_STOP   = 1.5    # initial stop = entry ∓ 1.5 × daily ATR (chandelier takes over)
# Inverse-vol (equal-risk) position sizing — the managed-futures standard: size
# each name to a common annualized-vol budget so a high-vol name (MARA) and a
# low-vol name (KO) contribute similar risk, instead of equal dollars. The
# multiplier scales the tier-based size by (target_vol / realized_vol), clipped
# so a very-low-vol name can't lever up and a very-high-vol name keeps a floor.
_MOM_TARGET_VOL      = 0.30   # 30% annualized vol budget per position
_MOM_VOL_SCALAR_MIN  = 0.30
_MOM_VOL_SCALAR_MAX  = 1.25
# Targets are intentionally WIDE/aspirational — momentum_monitor exits on the
# chandelier trail / trend break, NOT a fixed target. They exist only so the
# card shows the let-it-run upside; nothing acts on them.
_MOMENTUM_T1_R       = 4.0
_MOMENTUM_T2_R       = 8.0


def _fire_momentum(sb: Client, ms, direction: str) -> None:
    """Fire a swing momentum signal (detector_source=TREND_MOMENTUM)."""
    strategy_type = "swing_trade"
    if _has_active_signal(sb, ms.ticker, strategy_type):
        return
    price, atr = ms.last_price, ms.atr
    if not price or atr <= 0:
        return

    if direction == "LONG":
        stop = round(price - _MOMENTUM_ATR_STOP * atr, 2)
        risk = price - stop
        t1, t2 = round(price + _MOMENTUM_T1_R * risk, 2), round(price + _MOMENTUM_T2_R * risk, 2)
    else:
        stop = round(price + _MOMENTUM_ATR_STOP * atr, 2)
        risk = stop - price
        t1, t2 = round(price - _MOMENTUM_T1_R * risk, 2), round(price - _MOMENTUM_T2_R * risk, 2)
    if risk <= 0:
        return

    entry_gate_log: dict = {}
    try:
        from engine import alpaca_client as _alpaca
        df_e = _alpaca.get_bars(ms.ticker, timeframe="1Hour", days=10)
        gate = entry_gate.check(ticker=ms.ticker, direction=direction,
                                strategy_type=strategy_type, df_entry=df_e,
                                price=price, entry_tf="1h", detector="TREND_MOMENTUM")
        entry_gate_log = dict(gate.gate_log)
        if not gate.allowed:
            logger.info(f"[runner] {ms.ticker} TREND_MOMENTUM blocked: {' | '.join(gate.reasons)}")
            entry_gate.log_rejection(sb=sb, ticker=ms.ticker, direction=direction,
                                     strategy_type=strategy_type, price=price,
                                     confidence_score=75, gate=gate, detector="TREND_MOMENTUM")
            return
    except Exception as e:
        logger.warning(f"[runner] {ms.ticker} TREND_MOMENTUM gate error (failing open): {e}")

    risk_chk = risk_manager.check(sb, ms.ticker, 75)
    if not risk_chk["allowed"]:
        logger.info(f"[runner] {ms.ticker} TREND_MOMENTUM blocked — portfolio: {risk_chk['block_reason']}")
        return

    # ── Inverse-vol (equal-risk) sizing ───────────────────────────────────────
    # Scale the tier-based position size to a common vol budget so each momentum
    # name carries comparable risk (managed-futures standard).
    vol_scalar = 1.0
    if ms.ann_vol and ms.ann_vol > 0:
        vol_scalar = max(_MOM_VOL_SCALAR_MIN,
                         min(_MOM_VOL_SCALAR_MAX, _MOM_TARGET_VOL / ms.ann_vol))
    pos_mult = round(risk_chk["position_mult"] * vol_scalar, 3)

    setup_reason = (f"Trend momentum — {direction} (rank z={ms.score:+.2f}, "
                    f"12-1 mom {ms.raw_return * 100:+.1f}% @ {ms.ann_vol * 100:.0f}% vol, "
                    f"size {vol_scalar:.2f}x inv-vol)")
    logger.info(f"[runner] {ms.ticker} TREND_MOMENTUM {direction} @ ${price:.2f}  "
                f"z={ms.score:+.2f}  inv-vol={vol_scalar:.2f}x")
    signal_row = {
        "ticker":             ms.ticker,
        "direction":          direction,
        "entry_price":        round(price, 2),
        "stop_loss":          stop,
        "target_one":         t1,
        "target_two":         t2,
        "confidence_score":   75,
        "confidence_factors": [setup_reason],
        "timeframe":          "1d",
        "strategy_type":      strategy_type,
        "status":             "active",
        "ai_explanation":     setup_reason,
        "regime_type":        _live_regime_type(),
        "session_mode":       "",
        "confidence_tier":    risk_chk["confidence_tier"],
        "position_multiplier": pos_mult,
        "risk_reward":        _MOMENTUM_T1_R,
        "score_breakdown":    {
            "detector_source":   "TREND_MOMENTUM",
            "predictive_setup":  "TREND_MOMENTUM",
            "predictive_reason": setup_reason,
            "mom_score":         ms.score,
            "blended_return":    ms.raw_return,
            "ann_vol":           ms.ann_vol,
            "vol_target":        _MOM_TARGET_VOL,
            "vol_scalar":        round(vol_scalar, 3),
            "base_position_mult": risk_chk["position_mult"],
            "sma_fast":          ms.sma_fast,
            "sma_slow":          ms.sma_slow,
            "ext_atr":           getattr(ms, "ext_atr", 0.0),
            "entry_gate":        entry_gate_log,
        },
        "confidence_grade":   "B+",
        "setup_type":         "TREND_MOMENTUM",
    }
    new_id = _write_signal(sb, signal_row)
    try:
        push.send_signal_alert(ms.ticker, direction, 75, "stock",
                               signal_id=str(new_id) if new_id else None)
    except Exception:
        pass


def _run_momentum_scan() -> None:
    """Daily cross-sectional momentum scan: rank the universe by vol-adjusted
    trend momentum, fire the strongest uptrend names LONG (and the weakest
    downtrend names SHORT only in a bearish regime). Scheduled ~10 AM ET."""
    from engine.session_classifier import is_market_open_today
    if not is_market_open_today():
        logger.info("[runner] Momentum scan skipped — market closed today")
        return
    logger.info("[runner] ═══ Momentum scan started ═══")
    try:
        from engine import momentum_detector as md, alpaca_client as _alpaca
        sb = _supabase()
        try:
            regime = regime_detector.detect()
        except Exception:
            regime = {"regime_type": "RANGING"}
        bearish = regime.get("regime_type") in ("TRENDING_BEAR", "RISK_OFF", "PANIC")

        scores = []
        for tk in md.UNIVERSE:
            try:
                # ~500 calendar days ≈ 345 trading bars — enough for the canonical
                # 12-1 formation (needs 252 + 21 = 273 bars) plus margin.
                df = _alpaca.get_bars(tk, timeframe="1Day", days=500)
                ms = md.score(tk, df)
                if ms and ms.bias != "NONE":
                    scores.append(ms)
            except Exception:
                continue

        # ── Chase-guard: skip top-ranked names that are a poor ENTRY today —
        # stretched into a blow-off OR rolling over short-term (the TXN case).
        # Filter BEFORE taking top-N so non-chasing leaders fill the slots.
        def _chasing(s) -> bool:
            ext = getattr(s, "ext_atr", 0.0)
            if s.bias == "LONG":
                return ext > md._CHASE_MAX_ATR or ext < md._CHASE_MIN_ATR
            return ext < -md._CHASE_MAX_ATR or ext > -md._CHASE_MIN_ATR
        for s in [x for x in scores if x.bias in ("LONG", "SHORT") and _chasing(x)]:
            logger.info(f"[runner] momentum CHASE-SKIP {s.ticker} {s.bias} "
                        f"ext={getattr(s,'ext_atr',0):+.1f} ATR vs EMA{md._CHASE_EMA} "
                        f"(z={s.score:+.2f}) — poor entry, waiting for pullback")

        longs  = sorted([s for s in scores if s.bias == "LONG" and not _chasing(s)],
                        key=lambda x: -x.score)[:_MOMENTUM_MAX_LONGS]
        shorts = (sorted([s for s in scores if s.bias == "SHORT" and not _chasing(s)],
                         key=lambda x: x.score)[:_MOMENTUM_MAX_SHORTS]
                  if bearish else [])
        for s in longs:
            _fire_momentum(sb, s, "LONG")
        for s in shorts:
            _fire_momentum(sb, s, "SHORT")
        logger.info(f"[runner] ═══ Momentum scan done — {len(scores)} qualified, "
                    f"fired {len(longs)} longs / {len(shorts)} shorts (regime={regime.get('regime_type')}) ═══")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[runner] Momentum scan failed: {e}", exc_info=True)


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

    # ── Batch price fetch: one Alpaca call for ALL active tickers ─────────────
    # Include EXPIRED tickers too, so we can record the expiry close price + P/L
    # (an expired signal was previously closed with no price/result — DVN showed
    # grey at +2.5% with no info, 2026-05-28).
    all_tickers = [sig["ticker"] for sig in rows]

    price_map: dict[str, float] = {}
    if all_tickers:
        # Deduplicate
        unique_tickers = list(dict.fromkeys(all_tickers))
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
        # TREND_MOMENTUM is managed SOLELY by engine.momentum_monitor (chandelier
        # trail, structural SMA50 backstop, daily-close trend-break exit, NO fixed
        # target and NO time expiry — let winners run). The generic closer must
        # skip it, otherwise it would (a) force-expire the trade at the swing
        # 10-day backstop and (b) stop it out on an intraday wick against a stale
        # stop — both contradict the trend-following design. (signal_monitor
        # already skips it for the same reason.)
        if ((sig.get("score_breakdown") or {}).get("detector_source")) == "TREND_MOMENTUM":
            continue
        # MANUAL override: the admin — or a long-horizon signal like deep_value —
        # owns this trade. The engine must NOT auto-expire or stop/target-close it
        # (signal_monitor + the RT path already skip manual; the expiry must too,
        # else a manual / months-hold position is force-closed at its strategy's
        # max-hold default).
        if (sig.get("management_mode") or "engine") == "manual":
            continue

        created      = datetime.fromisoformat(sig["created_at"].replace("Z", "+00:00"))
        strategy     = sig.get("strategy_type") or "day_trade"
        reason: Optional[str] = None
        close_price: Optional[float] = None

        if is_past_max_hold(created, strategy):
            reason = "expired"
            close_price = price_map.get(sig["ticker"])   # record P/L at expiry
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
                # Record the close price + P/L even on expiry so the card isn't
                # a blank grey close (DVN expired at +2.5% with no info).
                if close_price is not None:
                    entry   = float(sig["entry_price"])
                    is_long = sig["direction"] == "LONG"
                    raw_pct = ((close_price - entry) / entry) * 100 if is_long \
                              else ((entry - close_price) / entry) * 100
                    update["result_pct"] = round(raw_pct, 4)
                    update["result_pnl"] = round((close_price - entry) if is_long
                                                  else (entry - close_price), 4)
            elif close_price is not None:
                entry   = float(sig["entry_price"])
                is_long = sig["direction"] == "LONG"
                raw_pct = ((close_price - entry) / entry) * 100 if is_long \
                          else ((entry - close_price) / entry) * 100
                raw_pnl = (close_price - entry) if is_long else (entry - close_price)
                # Classify by P&L SIGN, not by which level was hit. A trailing
                # stop raised above entry closes 'stop_hit' but IN PROFIT — that's
                # a WIN (fixes the 'ON +1.47% loss' corruption, 2026-06-04).
                update["result"] = result_from_pnl_pct(raw_pct)
                if reason == "target_hit":
                    hit_t2 = (is_long and close_price >= sig["target_two"]) or \
                             (not is_long and close_price <= sig["target_two"])
                    update["hit_target"] = "t2" if hit_t2 else "t1"
                else:
                    update["hit_target"] = "sl"
                update["result_pct"] = round(raw_pct, 4)
                update["result_pnl"] = round(raw_pnl, 4)
            try:
                sb.table("signals").update(update).eq("id", sig["id"]).execute()
                logger.info(f"[closer] CLOSED stock {sig['ticker']} [{strategy}] ({reason})")
                # Log a close event so the signal history shows the exit price +
                # time (the batch closer previously updated the row silently —
                # SIDU stopped out with no 'stopped out @ $X' entry, 2026-05-28).
                if close_price is not None:
                    _pct = update.get("result_pct")
                    _ptxt = f" ({_pct:+.1f}%)" if _pct is not None else ""
                    if reason == "stop_hit":
                        ev, emo, label = "closed_loss", "🛑", "Stopped out"
                    elif reason == "target_hit":
                        ev, emo, label = "closed_win", "✅", "Target hit"
                    else:
                        ev, emo, label = "expired", "⏳", "Expired"
                    sb.table("signal_events").insert({
                        "signal_id":  sig["id"],
                        "event_type": ev,
                        "price":      round(float(close_price), 4),
                        "note":       f"{emo} {label} @ ${close_price:.2f}{_ptxt}",
                    }).execute()
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
        # MANUAL: months-horizon option holds (deep-value LEAPS) — engine
        # hands-off. Without this skip the 24h cutoff below would force-"expire"
        # a 1-2yr LEAP the next day (same class as the stock-expiry manual fix).
        if (sig.get("management_mode") or "engine") == "manual":
            continue
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


def _run_deep_value_signal() -> None:
    """Crash/deep-value long-term BUY combine (backlog #10). Regime-gated → no-ops
    in a healthy market; fires manual-mode position signals for quality names that
    are deeply discounted AND showing a confirmed turn (falling-knife guard)."""
    try:
        from engine import deep_value_signal
        sb = create_client(os.environ["SUPABASE_URL"], _supabase_key())
        res = deep_value_signal.generate(sb)
        logger.info(f"[runner] ═══ Deep-value combine — {res} ═══")
    except Exception as e:
        logger.error(f"[runner] Deep-value combine failed: {e}", exc_info=True)


def _run_drawdown_regime_log() -> None:
    """Daily ops log of the broad-market drawdown regime (Phase 0 of the
    crash/deep-value signal). Visibility only — surfacing/alerting is a later step."""
    try:
        from engine import drawdown_regime
        r = drawdown_regime.assess(force=True)
        logger.info(f"[runner] ═══ Drawdown regime: {r['regime']} "
                    f"(SPY {r.get('off_high_pct')}% off high) — "
                    f"accumulation_window={r['accumulation_window']} ═══")
    except Exception as e:
        logger.debug(f"[runner] drawdown regime log failed: {e}")


def _run_fundamentals_refresh() -> None:
    """Rolling refresh of the EDGAR fundamentals quality screen — a stale batch
    each run. Fundamentals change quarterly, so a few runs/day covers the whole
    universe well within cadence. Best-effort; no-ops if the cache table is absent."""
    try:
        from engine import fundamentals
        sb = create_client(os.environ["SUPABASE_URL"], _supabase_key())
        res = fundamentals.refresh_universe(sb, batch=20)
        logger.info(f"[runner] ═══ Fundamentals refresh — {res} ═══")
    except Exception as e:
        logger.error(f"[runner] Fundamentals refresh failed: {e}", exc_info=True)


def _run_phantom_audit() -> None:
    """
    EOD data-integrity audit. Verifies every signal closed today against the
    real 1-min tape so a bad-price / phantom close (the 2026-06-03 incident)
    can't silently rot the track record. Logs a summary and pushes an
    ADMIN-ONLY alert (clean or flagged). Runs ~4:50 PM ET, post-close.
    """
    try:
        from engine import phantom_audit
        res = phantom_audit.run_and_alert(days=1)
        logger.info(f"[runner] ═══ Phantom audit done — {res['audited']} audited, "
                    f"{res['serious_count']} serious, {res['flagged_count']} flagged ═══")
        return (f"{res['audited']} audited · {res['serious_count']} serious · "
                f"{res['flagged_count']} flagged")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[runner] Phantom audit failed: {e}", exc_info=True)


def _run_gate_validator() -> None:
    """
    Nightly entry-gate rejection validator. Walks unjudged rows in
    entry_gate_rejections and backfills would_have_won + realized_pnl_pct
    via historical bar simulation. Runs daily at 3 AM UTC.
    """
    logger.info("[runner] ═══ Entry-gate rejection validator started ═══")
    sb = None
    try:
        from engine import gate_validator
        sb = create_client(os.environ["SUPABASE_URL"], _supabase_key())
        result = gate_validator.validate_batch(sb, limit=500)
        logger.info(f"[runner] ═══ Validator done — {result} ═══")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[runner] Gate validator failed: {e}", exc_info=True)

    # Armed-zone counterfactual: judge unfired zones (would a breakout have won?)
    try:
        from engine import zone_validator
        if sb is None:
            sb = create_client(os.environ["SUPABASE_URL"], _supabase_key())
        zresult = zone_validator.validate_batch(sb, limit=500)
        logger.info(f"[runner] ═══ Zone validator done — {zresult} ═══")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[runner] Zone validator failed: {e}", exc_info=True)

    # Breakout-watch: judge watched setups (broke out & followed through?) → Watch Accuracy
    try:
        from engine import breakout_validator
        if sb is None:
            sb = create_client(os.environ["SUPABASE_URL"], _supabase_key())
        bresult = breakout_validator.judge_batch(sb, limit=500)
        logger.info(f"[runner] ═══ Breakout-watch validator done — {bresult} ═══")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[runner] Breakout-watch validator failed: {e}", exc_info=True)


def _run_momentum_monitor() -> None:
    """Post-close manager for the systematic momentum model (chandelier trail +
    daily-close trend-break exit). Self-contained — generic signal_monitor
    skips TREND_MOMENTUM."""
    logger.info("[runner] ═══ Momentum monitor started ═══")
    try:
        from engine import momentum_monitor
        sb = create_client(os.environ["SUPABASE_URL"], _supabase_key())
        result = momentum_monitor.manage(sb)
        logger.info(f"[runner] ═══ Momentum monitor done — {result} ═══")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[runner] Momentum monitor failed: {e}", exc_info=True)


def _clear_zones_overnight() -> None:
    """Wipe all armed per-tick zones overnight (12:30 AM ET). Zones are kept
    through after-hours for admin analysis and cleared here so the next
    session arms fresh."""
    logger.info("[runner] ═══ Overnight armed-zone clear started ═══")
    try:
        # Close out any zones that armed but never fired before wiping memory,
        # so the history table reflects their 'expired' outcome.
        try:
            from engine import zone_history as _zh
            sb = create_client(os.environ["SUPABASE_URL"], _supabase_key())
            _zh.expire_stale(sb)
        except Exception as e:
            logger.debug(f"[runner] zone_history expire_stale skipped: {e}")
        from engine import stream as _stream
        _stream.clear_all_zones()
        logger.info("[runner] ═══ Overnight armed-zone clear done ═══")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[runner] Overnight zone clear failed: {e}", exc_info=True)


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

    # Per-job run ledger — records last-run/status/duration of EVERY job below to
    # the job_runs table (one listener, no per-job edits) for the Daily Jobs report.
    try:
        from engine import job_runs as _job_runs
        _job_runs.attach(scheduler, _supabase)
    except Exception as _jr_e:
        logger.debug(f"[runner] job_runs ledger not attached: {_jr_e}")

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

    # ── Breakout-watch lifecycle sync (every 5 min, RTH) ────────────────────
    # Maintains breakout_watch_history episodes (enter / trigger / fade / expire)
    # off the live Quant breakout bucket — gives the dashboard history + powers
    # Watch Accuracy. One row per watch EPISODE, not per refresh.
    def _run_breakout_watch():
        from zoneinfo import ZoneInfo as _ZI
        from datetime import datetime as _dt
        _now = _dt.now(_ZI("America/New_York"))
        _mins = _now.hour * 60 + _now.minute
        if not (570 <= _mins <= 960 and _now.weekday() < 5):   # 9:30 AM–4:00 PM ET
            return
        # (dashboardKey, direction, needs_trigger, levelField, scoreField, stageField, stageValue)
        # stageField/stageValue: only track episodes at the CONFIRMED stage. For
        # the cycle buckets we record ONLY the Buy Zone / Peak (not the Watch/base
        # stage) — measuring a Watch on ±8% barriers would just capture base noise
        # (HOOD chopping 70↔76↔63). Episodes anchor at the confirmed turn.
        _BUCKETS = [
            ("breakouts",      "up",   True,  "breakoutLevel",  "breakoutScore",      None,              None),
            ("breakdowns",     "down", True,  "breakdownLevel", "breakdownScore",     None,              None),
            ("topMomentum",    "up",   False, None,             "momentumScore",      None,              None),
            ("pullbacks",      "up",   False, None,             "finalQuantScore",    None,              None),
            ("highVolumeUp",   "up",   False, None,             "volumeScore",        None,              None),
            ("highVolumeDown", "down", False, None,             "volumeScore",        None,              None),
            ("vwapReclaim",    "up",   False, None,             "finalQuantScore",    None,              None),
            ("oversoldBounce", "up",   False, None,             "meanReversionScore", None,              None),
            ("turnaround",     "up",   False, None,             "turnaroundScore",    "turnaroundStage", "buyzone"),
            ("peak",           "down", False, None,             "peakScore",          "peakStage",       "peak"),
        ]
        try:
            from engine import quant_score_service, breakout_watch
            dash = quant_score_service.get_quant_dashboard() or {}
            sb = _supabase()
            for _key, _dir, _needs, _lvl_f, _score_f, _stage_f, _stage_v in _BUCKETS:
                rows = [
                    {"ticker": r.get("ticker"), "price": r.get("price"),
                     "level": (r.get(_lvl_f) if _lvl_f else None),
                     "score": r.get(_score_f)}
                    for r in (dash.get(_key) or [])
                    if (_stage_f is None or r.get(_stage_f) == _stage_v)
                ]
                if rows:
                    breakout_watch.sync_watch(sb, rows, bucket=_key,
                                              direction=_dir, needs_trigger=_needs)

            # ── Cycle Buy-Zone / Peak push alerts (watchlist-scoped, deduped) ──
            from engine import cache as _cache
            from datetime import datetime as _dt3, timezone as _tz3
            _today = _dt3.now(_tz3.utc).date().isoformat()
            _alerts = [(r.get("ticker"), "turnaround")
                       for r in (dash.get("turnaround") or [])
                       if r.get("turnaroundStage") == "buyzone"]
            _alerts += [(r.get("ticker"), "peak")
                        for r in (dash.get("peak") or [])
                        if r.get("peakStage") == "peak"]
            for _tk, _kind in _alerts[:8]:
                if not _tk:
                    continue
                _ck = f"cycle_alert:{_kind}:{_tk}:{_today}"
                try:
                    if _cache.kv.get_json(_ck):
                        continue
                except Exception:
                    pass
                _sent = push.send_cycle_alert(_tk, _kind, sb=sb)
                # Dedup per ticker/kind/day REGARDLESS of how many users were
                # pushed — send_cycle_alert always records the alert row, so
                # without this a name with no watchers (sent=0) re-records every
                # 5-min scan (the TWLO×7/hr feed spam). Set the day-key always.
                try:
                    _cache.kv.set_json(_ck, {"sent": _sent}, 86400)
                except Exception:
                    pass
                if _sent:
                    logger.info(f"[runner] cycle alert sent: {_tk} {_kind} ({_sent} users)")

            # ── PEAK_FORMING / TURN_FORMING — EARLY anticipatory tracked cards ──
            # Fire off the cycle 'watch' stage (topping / bottoming FORMING, before
            # the confirmed Peak / Buy-Zone) so we catch the swing earlier. Separate
            # measured experiments; deduped + sized down inside forming_signals.
            try:
                from engine import forming_signals as _fs
                _peak_watch = [r for r in (dash.get("peak") or [])
                               if r.get("peakStage") == "watch"]
                _turn_watch = [r for r in (dash.get("turnaround") or [])
                               if r.get("turnaroundStage") == "watch"]
                _peak_watch.sort(key=lambda r: -(r.get("peakScore") or 0))
                _turn_watch.sort(key=lambda r: -(r.get("turnaroundScore") or 0))
                _fg = 0
                for _r in _peak_watch[:3]:
                    if _fs.generate(sb, _r, "peak").get("stock"):
                        _fg += 1
                for _r in _turn_watch[:3]:
                    if _fs.generate(sb, _r, "turn").get("stock"):
                        _fg += 1
                if _fg:
                    logger.info(f"[runner] forming cycle cards generated: {_fg}")
            except Exception as _fe:
                logger.debug(f"[runner] forming cycle gen skipped: {_fe}")
        except Exception as _e:
            logger.error(f"[runner] setup-watch sync failed: {_e}")

    scheduler.add_job(
        _run_breakout_watch,
        trigger=IntervalTrigger(minutes=5),
        id="breakout_watch_sync",
        name="Breakout-watch lifecycle sync (5-min, RTH)",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled breakout-watch lifecycle sync every 5 min (RTH)")

    # ── Quant dashboard precompute (every 3 min, 24/7) ──────────────────────
    # The web endpoint serves the Redis-cached result; this keeps it warm so the
    # UI never crunches ~150 names on the request path (was timing out →
    # "engine unreachable"). Runs on the worker; web reads Redis.
    def _run_quant_refresh():
        try:
            from engine import quant_score_service
            quant_score_service.get_quant_dashboard(force=True)
        except Exception as _e:
            logger.error(f"[runner] quant dashboard refresh failed: {_e}")

    scheduler.add_job(
        _run_quant_refresh,
        trigger=IntervalTrigger(minutes=3),
        id="quant_refresh",
        name="Quant dashboard precompute (3-min)",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=20),
    )
    logger.info("[runner] Scheduled quant dashboard precompute every 3 min")

    # ── Community insights precompute (every 5 min, 24/7) ───────────────────
    # Same fix as quant: the /community/* enrichment (bars + manipulation + news
    # for ~30 names) was computed on the request path → timeouts. Precompute on
    # the worker so the endpoints serve the warm Redis cache instantly.
    def _run_community_refresh():
        try:
            from engine import social_insights
            sb = _supabase()
            social_insights.get_enriched_trending(sb, force=True)
            social_insights.community_pulse(sb, force=True)
            social_insights.whats_changed(sb, force=True)
        except Exception as _e:
            logger.error(f"[runner] community refresh failed: {_e}")

    scheduler.add_job(
        _run_community_refresh,
        trigger=IntervalTrigger(minutes=5),
        id="community_refresh",
        name="Community insights precompute (5-min)",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=40),
    )
    logger.info("[runner] Scheduled community insights precompute every 5 min")

    # ── Watchlist state-change alerts — every 15 min on trading days ──────
    # Pushes a user when one of their watched tickers flips state (buy zone /
    # topping / breakout / lost trend). Seeds baseline on first sight + per-day
    # dedup so it never spams. Skips non-trading days.
    def _run_watchlist_alerts():
        try:
            # Regular trading hours only (9:30 AM ET → close) — not pre-market,
            # after-hours, or overnight. Quant reads are stale when the tape is
            # closed and users shouldn't get pushes at 2 AM.
            from engine.session_classifier import is_market_open_now
            if not is_market_open_now():
                return
            from engine import watchlist_alerts
            watchlist_alerts.run(_supabase())
        except Exception as _e:
            logger.error(f"[runner] watchlist alerts failed: {_e}")

    scheduler.add_job(
        _run_watchlist_alerts,
        trigger=IntervalTrigger(minutes=15),
        id="watchlist_alerts",
        name="Watchlist state-change alerts (15-min)",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=90),
    )
    logger.info("[runner] Scheduled watchlist state-change alerts every 15 min")

    # ── Breakdown / heavy-selling alerts (universe-wide, 15-min) ─────────────
    # Two-stage broadcast: EARLY (lost 20-day avg on heavy down-vol) + CONFIRMED
    # (broke 20-day low on vol). Reuses the cached full quant scan; per-ticker
    # transition state + per-day dedup + a hard per-run cap keep it from spamming.
    def _run_breakdown_alerts():
        try:
            # Regular trading hours ONLY — breakdown SIGNALS (short/put cards) are
            # generated here, and you can't short / buy puts when the market is
            # closed, plus options_scanner needs a live chain. No pre/post/overnight.
            from engine.session_classifier import is_market_open_now
            if not is_market_open_now():
                return
            from engine import breakdown_alerts
            breakdown_alerts.run(_supabase())
        except Exception as _e:
            logger.error(f"[runner] breakdown alerts failed: {_e}")

    scheduler.add_job(
        _run_breakdown_alerts,
        trigger=IntervalTrigger(minutes=15),
        id="breakdown_alerts",
        name="Breakdown / heavy-selling alerts (15-min)",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=150),
    )
    logger.info("[runner] Scheduled breakdown alerts every 15 min")

    # ── Breakout / unusual-buying alerts (universe-wide, 15-min) ─────────────
    # Bullish mirror: EARLY (pressing 20-day high on up-vol) + CONFIRMED (broke
    # the high → also generates LONG + CALL cards) + ACCUMULATION (heavy buying).
    def _run_breakout_alerts():
        try:
            from engine.session_classifier import is_market_open_now
            if not is_market_open_now():
                return
            from engine import breakout_alerts
            breakout_alerts.run(_supabase())
        except Exception as _e:
            logger.error(f"[runner] breakout alerts failed: {_e}")

    scheduler.add_job(
        _run_breakout_alerts,
        trigger=IntervalTrigger(minutes=15),
        id="breakout_alerts",
        name="Breakout / unusual-buying alerts (15-min)",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=180),
    )
    logger.info("[runner] Scheduled breakout alerts every 15 min")

    # ── Premarket disaster-gap alerts (notification-only, 8:00–9:30 AM ET) ────
    # Heads-up for OPEN overnight positions (swing/breakout/breakdown/TREND) that
    # gapped hard AGAINST the signal before the open. NEVER closes a position or
    # records a result on a premarket print (thin/wicky, options shut, often
    # reverses by 9:30) — purely "watch the open". The module gates itself to the
    # 8:00–9:30 AM ET window (no earlier — respects the no-overnight-push rule)
    # with per-signal-per-day dedup.
    def _run_premarket_alerts():
        try:
            from engine import premarket_alerts
            premarket_alerts.run(_supabase())
        except Exception as _e:
            logger.error(f"[runner] premarket alerts failed: {_e}")

    scheduler.add_job(
        _run_premarket_alerts,
        trigger=IntervalTrigger(minutes=15),
        id="premarket_alerts",
        name="Premarket disaster-gap alerts (8:00–9:30 AM ET)",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=210),
    )
    logger.info("[runner] Scheduled premarket disaster-gap alerts every 15 min (8:00–9:30 AM ET gate)")

    # ── Market-regime timeline (write-on-change) ─────────────────────────────
    # Evaluate the regime every 5 min across 4:00 AM–8:00 PM ET and append a row
    # to regime_history ONLY when regime_type/session flips — so we keep the day's
    # transitions (pre → PANIC → bull → chop), enabling intraday regime-at-fire /
    # regime-during-hold analysis. No-op overnight/weekends.
    def _run_regime_capture():
        try:
            from zoneinfo import ZoneInfo as _ZI
            et = datetime.now(_ZI("America/New_York"))
            if et.weekday() >= 5:
                return
            mins = et.hour * 60 + et.minute
            if not (4 * 60 <= mins <= 20 * 60):   # 4:00 AM – 8:00 PM ET only
                return
            from engine import regime_history
            regime_history.record_if_changed(_supabase())
        except Exception as _e:
            logger.error(f"[runner] regime capture failed: {_e}")

    scheduler.add_job(
        _run_regime_capture,
        trigger=IntervalTrigger(minutes=5),
        id="regime_capture",
        name="Market-regime timeline (write-on-change, 4 AM–8 PM ET)",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=45),
    )
    logger.info("[runner] Scheduled regime-timeline capture every 5 min (write-on-change)")

    # ── Daily EOD performance snapshot (~8:05 PM ET) ─────────────────────────
    # One immutable row/day in daily_performance: today's CLOSED outcomes (by
    # detector / conviction / direction), profit give-back (peak vs realized),
    # the regime path, and the ACTIVE-book state. Runs AFTER the full 4 AM–8 PM
    # extended session so the MFE peaks are complete (captures AH give-back).
    # Upsert is idempotent per trade_date; the 15-min interval + ET gate fires it
    # once in the 8:05–8:35 PM window.
    def _run_daily_performance():
        try:
            from zoneinfo import ZoneInfo as _ZI
            et = datetime.now(_ZI("America/New_York"))
            if et.weekday() >= 5:
                return
            mins = et.hour * 60 + et.minute
            if not (20 * 60 + 5 <= mins <= 20 * 60 + 35):   # 8:05–8:35 PM ET
                return
            from engine import daily_performance
            row = daily_performance.compute_and_store(_supabase())
            if row:
                # Lead with the HONEST equal-weight account return (avg/trade);
                # closed_net_pct is the SUM of trade %s, NOT an account return.
                return (f"{row.get('closed_n', 0)} closed · win {row.get('closed_win_rate')}% · "
                        f"acct {row.get('closed_avg_pct')}%/trade (equal-wt) · "
                        f"Σ {row.get('closed_net_pct')}% (sum, not a return) · "
                        f"active {row.get('active_n', 0)} · giveback {row.get('giveback_pct')}%")
        except Exception as _e:
            logger.error(f"[runner] daily performance snapshot failed: {_e}")

    scheduler.add_job(
        _run_daily_performance,
        trigger=IntervalTrigger(minutes=15),
        id="daily_performance",
        name="Daily EOD performance snapshot (~8:05 PM ET)",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=75),
    )
    logger.info("[runner] Scheduled daily EOD performance snapshot (~8:05 PM ET)")

    # ── Cycle tracked cards — turnaround (LONG+CALL) + peak (SHORT+PUT) ───────
    # The Buy-Zone / Peak PUSH alerts fire from the 5-min breakout-watch sync;
    # this generates the TRADEABLE cards for those same names. State-based +
    # deduped + capped, so a name that confirmed its turn overnight/pre-market
    # is captured at the RTH open (mirrors breakdown/breakout alerts).
    def _run_cycle_signals():
        try:
            # Regular trading hours ONLY — turnaround/peak SIGNALS (long/short +
            # option cards) are generated here; you can't act on them when the
            # market is closed, and options_scanner needs a live chain.
            from engine.session_classifier import is_market_open_now
            if not is_market_open_now():
                return
            from engine import cycle_signals
            cycle_signals.run(_supabase())
        except Exception as _e:
            logger.error(f"[runner] cycle signals failed: {_e}")

    scheduler.add_job(
        _run_cycle_signals,
        trigger=IntervalTrigger(minutes=15),
        id="cycle_signals",
        name="Cycle tracked cards — turnaround + peak (15-min)",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=210),
    )
    logger.info("[runner] Scheduled cycle signals every 15 min")

    # ── Community social snapshot (hourly, 24/7) ────────────────────────────
    # Captures the merged trending feed + price-at-capture into social_snapshots.
    # This time-series is what powers going-viral z-scores, buzz velocity,
    # mention sparklines, "what changed", and the trending->returns track record
    # on the Community tab. Runs hourly — social buzz is a 24/7 signal, not
    # RTH-bound. No-ops gracefully (logs) until the social_snapshots table exists.
    def _run_social_snapshot():
        try:
            from engine import social_sentiment
            from engine import alpaca_client as _alpaca
            data = social_sentiment.get_trending(limit=50, force=True) or {}
            rows = data.get("trending") or []
            if not rows:
                logger.info("[runner] social snapshot: no trending data, skipping")
                return
            tickers = [r.get("ticker") for r in rows if r.get("ticker")]
            prices  = _alpaca.get_latest_prices(tickers) or {}
            snap_rows = []
            for i, r in enumerate(rows):
                t = r.get("ticker")
                if not t:
                    continue
                snap_rows.append({
                    "ticker":              t,
                    "name":                r.get("name"),
                    "rank":                i + 1,
                    "score":               r.get("score"),
                    "reddit_mentions":     r.get("reddit_mentions"),
                    "reddit_rank":         r.get("reddit_rank"),
                    "reddit_sentiment":    r.get("reddit_sentiment"),
                    "stocktwits_rank":     r.get("stocktwits_rank"),
                    "stocktwits_watchers": r.get("stocktwits_watchers"),
                    "sources":             r.get("sources"),
                    "price":               prices.get(t),
                })
            sb = _supabase()
            sb.table("social_snapshots").insert(snap_rows).execute()
            logger.info(f"[runner] social snapshot captured {len(snap_rows)} tickers "
                        f"({len(prices)} priced)")

            # ── Buzz-spike push alerts (watchlist-scoped, deduped per day) ──
            # Only fires once a ticker has enough history for a real z-score, so
            # this is a no-op for the first several days after launch.
            try:
                from engine import social_insights, cache
                from datetime import datetime as _dt2, timezone as _tz2
                current = {r["ticker"]: r.get("reddit_mentions") for r in snap_rows}
                chg = {r.get("ticker"): r.get("reddit_change_pct") for r in rows}
                spikes = social_insights.detect_spikes(sb, list(current), current)
                today = _dt2.now(_tz2.utc).date().isoformat()
                for sp in spikes[:5]:   # cap dispatches per run
                    t = sp["ticker"]
                    dedup_key = f"buzz_alert:{t}:{today}"
                    try:
                        if cache.kv.get_json(dedup_key):
                            continue
                    except Exception:
                        pass
                    sent = push.send_buzz_spike_alert(
                        t, change_pct=chg.get(t), mentions=sp.get("mentions"), sb=sb)
                    if sent:
                        try:
                            cache.kv.set_json(dedup_key, {"sent": sent}, 86400)
                        except Exception:
                            pass
                        logger.info(f"[runner] buzz spike alert sent: {t} "
                                    f"(z={sp['z']}, {sent} users)")
            except Exception as _be:
                logger.error(f"[runner] buzz spike alerts failed: {_be}")
        except Exception as _e:
            logger.error(f"[runner] social snapshot failed: {_e}")

    scheduler.add_job(
        _run_social_snapshot,
        trigger=IntervalTrigger(hours=1),
        id="social_snapshot",
        name="Community social trending snapshot (hourly)",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled community social snapshot every 1h")

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

    # ── Post-close entry-gate rejection validator (4:15 PM ET / 3:15 PM CDT) ─
    # Runs just AFTER the close so the full RTH session is available. Combined
    # with the intraday forward-walk being capped at the 4 PM ET close
    # (gate_validator), this judges the ENTIRE day's intraday rejections in one
    # run instead of leaving the afternoon ones pending on an 8h window.
    # Timezone is America/New_York so it tracks DST automatically.
    scheduler.add_job(
        _run_gate_validator,
        trigger=CronTrigger(hour=16, minute=15, timezone="America/New_York"),
        id="gate_validator",
        name="SignalBolt entry-gate rejection validator",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled entry-gate validator (4:15 PM ET, post-close)")

    # ── EOD phantom-data audit — 4:50 PM ET ──────────────────────────────
    # Verify the day's closes against the real 1-min tape so bad-price/phantom
    # closes are caught same-day (not a month later). Admin-only alert.
    scheduler.add_job(
        _run_phantom_audit,
        trigger=CronTrigger(hour=16, minute=50, timezone="America/New_York"),
        id="phantom_audit",
        name="SignalBolt EOD phantom-data audit",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled phantom-data audit (4:50 PM ET, post-close)")

    # ── Fundamentals quality-screen rolling refresh ──────────────────────
    # 3×/day (off-market hours) refreshes a stale batch; the ~150-name universe
    # cycles within days, well inside the quarterly cadence fundamentals change.
    scheduler.add_job(
        _run_fundamentals_refresh,
        trigger=CronTrigger(hour="6,13,19", minute=37, timezone="America/New_York"),
        id="fundamentals_refresh",
        name="SignalBolt EDGAR fundamentals refresh",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled fundamentals refresh (6:37/13:37/19:37 ET, rolling batch)")

    # ── Drawdown-regime daily log (Phase 0 of crash/deep-value signal) ───
    scheduler.add_job(
        _run_drawdown_regime_log,
        trigger=CronTrigger(hour=16, minute=12, timezone="America/New_York"),
        id="drawdown_regime_log",
        name="SignalBolt drawdown-regime log",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled drawdown-regime log (4:12 PM ET)")

    # ── Deep-value combine (crash/deep-value long-term BUY signal) ───────
    # Daily, post-close. Regime-gated → no-ops until a real -20% drawdown, then
    # fires manual-mode position signals for quality + deeply-discounted + turning
    # names (the falling-knife guard).
    scheduler.add_job(
        _run_deep_value_signal,
        trigger=CronTrigger(hour=16, minute=22, timezone="America/New_York"),
        id="deep_value_signal",
        name="SignalBolt deep-value combine",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled deep-value combine (4:22 PM ET)")

    # ── Overnight armed-zone clear — 12:30 AM ET ─────────────────────────
    # Zones are no longer cleared at the 4PM close so the admin can analyze
    # after-hours zone data in the app. This job wipes them ~half past
    # midnight ET so the next session arms fresh. Uses ET timezone directly
    # so it tracks DST without manual UTC offset bookkeeping.
    scheduler.add_job(
        _clear_zones_overnight,
        trigger=CronTrigger(hour=0, minute=30, timezone="America/New_York"),
        id="clear_zones_overnight",
        name="SignalBolt overnight armed-zone clear",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled overnight armed-zone clear (12:30 AM ET)")

    # ── Systematic momentum / trend scan — 10:00 AM ET weekdays ──────────
    # Cross-sectional momentum is a slow (daily) signal; run once after the
    # open settles. Fires swing signals tagged TREND_MOMENTUM, side-by-side
    # with SMC so the scorecard can compare realized edge.
    scheduler.add_job(
        _run_momentum_scan,
        trigger=CronTrigger(day_of_week="mon-fri", hour=10, minute=0, timezone="America/New_York"),
        id="momentum_scan",
        name="SignalBolt systematic momentum scan",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled momentum scan (10:00 AM ET, Mon-Fri)")

    # ── Momentum trade manager — 4:25 PM ET (after the daily bar settles) ──
    # Self-contained chandelier trail + daily-close trend-break exit for the
    # TREND_MOMENTUM model. Runs once daily (trend management is a daily event).
    scheduler.add_job(
        _run_momentum_monitor,
        trigger=CronTrigger(day_of_week="mon-fri", hour=16, minute=25, timezone="America/New_York"),
        id="momentum_monitor",
        name="SignalBolt momentum trade manager",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled momentum monitor (4:25 PM ET, Mon-Fri)")

    # ── Chart-read agreement track record — 4:40 PM ET (after the daily bar) ──
    # Logs each universe ticker's TA-vs-Quant read (one row/ticker/day) and scores
    # snapshots whose horizon has elapsed, so we learn which method is right when
    # they disagree. Best-effort; no-ops until chart_read_log table exists.
    def _run_chart_read_log():
        try:
            from engine import chart_read, quant_score_service
            sb = _supabase()
            chart_read.log_snapshot(sb, quant_score_service.DEFAULT_TICKERS)
            chart_read.score_snapshots(sb)
        except Exception as e:
            logger.error(f"[runner] chart-read log/score failed: {e}")

    scheduler.add_job(
        _run_chart_read_log,
        trigger=CronTrigger(day_of_week="mon-fri", hour=16, minute=40, timezone="America/New_York"),
        id="chart_read_log",
        name="Chart-read agreement track record (4:40 PM ET)",
        replace_existing=True,
    )
    logger.info("[runner] Scheduled chart-read agreement log (4:40 PM ET, Mon-Fri)")

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
