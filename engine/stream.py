"""
Event-Driven Signal Stream (Alpaca WebSocket)
=============================================
All five strategies fire exactly when their bar closes — zero polling lag.
Active scalp signals are tracked in real-time and closed within 1 minute
of T1, T2, or SL being hit — not on the 15-minute maintenance cycle.

Signal lifecycle:
  FIRE  — 5-min bar closes → SMC pipeline → signal in DB + push notification
          Latency: 2-5 seconds after bar close

  CLOSE — every 1-min bar: high/low checked against T1/T2/SL for active
          scalp signals. Hit detected within 60 seconds max.
          (day_trade/swing still use the 15-min tracker — longer holds)

Scan boundaries:
  minute % 5  == 0  →  scalp scan for that specific ticker
  minute % 15 == 0  →  day_trade + options_flow + dark_pool (all tickers)
  minute      == 0  →  swing_trade (all tickers)

Deduplication: _last_15m_barrier / _last_1h_barrier ensure each boundary
fires exactly ONCE even though 27 tickers all deliver bars in the same minute.

APScheduler now only handles:
  - Maintenance (tracker + signal_monitor) every 15 min
  - Weekly weight optimization (Sunday 2 AM UTC)

Environment:
  ALPACA_API_KEY       required
  ALPACA_SECRET_KEY    required
  ALPACA_DATA_FEED     "sip" (default) | "iex" (free-tier fallback)
"""

import asyncio
import concurrent.futures
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import sentry_sdk

logger = logging.getLogger("signalbolt.stream")

ET = ZoneInfo("America/New_York")

# ── Scan executor ─────────────────────────────────────────────
# Scans are CPU + I/O bound (Alpaca REST + Supabase writes).
# max_workers=5 → one per strategy type running concurrently without blocking.
_scan_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=5,
    thread_name_prefix="sb_scan",
)

# ── Bar boundary deduplication ────────────────────────────────
# Multiple tickers deliver bars at the same minute. We only want to fire
# each strategy scan ONCE per boundary, not once per ticker.
# Value = minute-of-day key (0-1439) of the last boundary that was processed.
_last_15m_barrier: int = -1   # day_trade / options_flow / dark_pool
_last_1h_barrier:  int = -1   # swing_trade

# ── Scalp signal real-time tracker ────────────────────────────
# Active scalp signals are cached here so every 1-minute bar can check
# T1/T2/SL without a Supabase query on each of the 27 bar events per minute.
# Cache is refreshed every 60 seconds and invalidated whenever a signal closes.
#
# Structure: { ticker: { id, direction, entry_price, stop_loss,
#                         target_one, target_two } }
_scalp_cache: dict = {}
_scalp_cache_ts: float = 0.0
_SCALP_CACHE_TTL: float = 60.0   # seconds

# ── Real-time trade-tick level checker (ALL strategies) ────────
# Every trade tick from Alpaca hits on_trade(). We throttle to at most
# 1 level check per second per ticker to avoid flooding the executor,
# while still catching price crosses within ~1 second of them happening.
#
# Structure: { ticker: [sig, sig, ...] }  — all active non-scalp signals
# (scalping is already handled by _check_scalp_levels via bar high/low)
_rt_cache:    dict[str, list[dict]] = {}  # ticker → active signals list
_rt_cache_ts: float = 0.0
_RT_CACHE_TTL:      float = 60.0  # full refresh every 60 s
_RT_THROTTLE_S:     float = 1.0   # max one level-check per second per ticker
_rt_last_check: dict[str, float] = {}   # ticker → last monotonic check time

# ── Tick-triggered scalp scanner ────────────────────────────────────────────
# Instead of waiting for a 5-min bar close, run a full SMC scalp scan the
# moment a trade tick arrives (throttled per ticker so we don't stack scans).
# This lets signals fire within ~1 second of a setup forming — not 0-5 min later.
#
# _TICK_SCALP_THROTTLE_S = 300 s  (5 min) — same cadence as bar-close scans but
# unlocked from the bar clock, so they can fire mid-bar when the move starts.
_TICK_SCALP_THROTTLE_S: float = 300.0
_tick_scalp_last: dict[str, float] = {}   # ticker → last tick-scan monotonic time

# ── Dynamic ticker subscription ────────────────────────────────────────────
# App WS clients may subscribe to tickers beyond ALL_TICKERS (custom watchlists).
# These are dynamically added to the live Alpaca trade stream so every ticker
# gets tick-by-tick updates — no REST polling fallback required.
#
# Threading model:
#   _wss_ref / _on_trade_ref are written from run_stream() (background task)
#   and read from subscribe_extra_tickers() (FastAPI async context).
#   Simple reference assignments are GIL-atomic in CPython — safe to read/write
#   without a lock as long as we check `is None` before use.
_subscribed_tickers: set[str] = set()   # all tickers currently live on Alpaca
_pending_tickers:    set[str] = set()   # requested before stream connected
_wss_ref                      = None    # StockDataStream (set while stream is live)
_on_trade_ref                 = None    # stored handler for re-subscriptions


async def subscribe_extra_tickers(tickers: list[str]) -> None:
    """
    Dynamically subscribe additional tickers to the live Alpaca trade stream.

    Called from /ws/prices when a client subscribes to tickers not in ALL_TICKERS
    (e.g. custom watchlist symbols). Once subscribed, Alpaca starts pushing trade
    ticks for those symbols → price_store.update() → broadcast to WS clients.

    Safe to call at any time — if the stream is not yet connected, the tickers are
    queued and applied as soon as the stream comes up (or on the next reconnect).
    """
    global _subscribed_tickers, _pending_tickers, _wss_ref, _on_trade_ref

    new = [t for t in tickers if t not in _subscribed_tickers]
    if not new:
        return

    _subscribed_tickers.update(new)

    if _wss_ref is None or _on_trade_ref is None:
        _pending_tickers.update(new)
        logger.debug(f"[stream] Queued dynamic tickers (stream not ready yet): {new}")
        return

    # subscribe_trades() calls asyncio.run_coroutine_threadsafe().result()
    # internally — blocks the calling thread, not the FastAPI event loop.
    # Run it in an executor so we don't block the async event loop.
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _wss_ref.subscribe_trades, _on_trade_ref, *new)
        logger.info(f"[stream] ✅ Dynamic trade subscription added: {new}")
    except Exception as e:
        # Stream may be mid-reconnect — pending set ensures retry on next connect
        _pending_tickers.update(new)
        logger.warning(f"[stream] Dynamic subscription deferred (will retry on reconnect): {e}")


# ── Context cache ─────────────────────────────────────────────
# Regime/session detection hits yfinance + Alpaca REST — expensive.
# Cache 4 minutes. All bar events within the same scan window share one fetch.
_regime_cache:  tuple[Optional[dict], float] = (None, 0.0)
_session_cache: tuple[Optional[dict], float] = (None, 0.0)
CONTEXT_TTL = 240   # seconds


def _get_regime() -> dict:
    global _regime_cache
    val, ts = _regime_cache
    if val is None or (time.monotonic() - ts) > CONTEXT_TTL:
        try:
            from engine import regime_detector
            val = regime_detector.detect()
            _regime_cache = (val, time.monotonic())
            logger.debug(f"[stream] Regime refreshed: {val['regime_type']} VIX={val['vix']}")
        except Exception as e:
            logger.warning(f"[stream] Regime refresh failed: {e} — using last known")
            val = val or {
                "regime_type": "RANGING", "vix": 18.0, "vix_change_pct": 0.0,
                "above_200ma": True, "adx": 20.0, "blocked": False, "block_reason": "",
            }
    return val


def _get_session() -> dict:
    global _session_cache
    val, ts = _session_cache
    if val is None or (time.monotonic() - ts) > CONTEXT_TTL:
        try:
            from engine import session_classifier
            val = session_classifier.classify()
            _session_cache = (val, time.monotonic())
            logger.debug(f"[stream] Session refreshed: {val['mode']}")
        except Exception as e:
            logger.warning(f"[stream] Session refresh failed: {e} — using last known")
            val = val or {
                "mode": "STANDARD", "market_open": True, "blocked": False,
                "block_reason": "", "threshold": 70, "sl_adjustment": 1.0,
                "allows_swing": True, "is_opex_day": False, "is_opex_week": False,
            }
    return val


# ── Scalp real-time close tracker ────────────────────────────

def _refresh_scalp_cache() -> None:
    """
    Refresh the in-memory cache of active scalp signals from Supabase.
    Called at most once per _SCALP_CACHE_TTL seconds (60s default).
    A fresh DB query per bar event (27/min) would be excessive.
    """
    global _scalp_cache, _scalp_cache_ts
    try:
        import os
        from supabase import create_client
        key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
        sb  = create_client(os.environ["SUPABASE_URL"], key)
        rows = (
            sb.table("signals")
            .select("id, ticker, direction, entry_price, stop_loss, target_one, target_two")
            .eq("status", "active")
            .eq("strategy_type", "scalping")
            .eq("result", "pending")
            .execute()
            .data
        ) or []
        _scalp_cache    = {r["ticker"]: r for r in rows}
        _scalp_cache_ts = time.monotonic()
        if rows:
            logger.debug(f"[stream] Scalp cache refreshed: {list(_scalp_cache.keys())}")
    except Exception as e:
        logger.debug(f"[stream] Scalp cache refresh failed: {e}")


def _close_scalp_signal(sig: dict, hit: str, bar_price: float) -> None:
    """
    Write close result to Supabase and send push notification.
    Called from the bar handler when a scalp T1/T2/SL level is breached.

    hit: "t1" | "t2" | "sl"
    bar_price: the bar's high or low that crossed the level
    """
    global _scalp_cache
    try:
        import os
        from supabase import create_client
        from datetime import datetime, timezone
        from engine import push

        entry    = float(sig["entry_price"])
        is_long  = sig["direction"] == "LONG"
        result   = "win" if hit in ("t1", "t2") else "loss"
        hit_price = float(sig["target_one"] if hit == "t1" else
                          sig["target_two"] if hit == "t2" else
                          sig["stop_loss"])

        pnl_pct = ((hit_price - entry) / entry * 100) if is_long else \
                  ((entry - hit_price) / entry * 100)
        pnl_abs = hit_price - entry if is_long else entry - hit_price

        update = {
            "status":        "closed",
            "result":        result,
            "hit_target":    hit,
            "result_pct":    round(pnl_pct, 4),
            "result_pnl":    round(pnl_abs, 4),
            "closed_reason": "target_hit" if result == "win" else "stop_hit",
            "closed_at":     datetime.now(timezone.utc).isoformat(),
        }

        key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
        sb  = create_client(os.environ["SUPABASE_URL"], key)
        sb.table("signals").update(update).eq("id", sig["id"]).execute()

        # Log to signal_events timeline
        try:
            hit_map   = {"t1": "Target 1", "t2": "Target 2", "sl": "Stop Loss"}
            hit_label = hit_map.get(hit, hit.upper())
            note = (
                f"{hit_label} hit @ ${hit_price:.2f} — "
                f"{'closed +' if result == 'win' else 'stopped out '}{abs(pnl_pct):.1f}%"
            )
            sb.table("signal_events").insert({
                "signal_id":  sig["id"],
                "event_type": "closed_win" if result == "win" else "closed_loss",
                "price":      hit_price,
                "note":       note,
            }).execute()
        except Exception:
            pass

        # Push notification
        try:
            ticker = sig["ticker"]
            if result == "win":
                push._send_raw(
                    title=f"[{'+' if result == 'win' else ''}] Scalp {hit.upper()} Hit - {ticker}  +{pnl_pct:.1f}%",
                    body=f"{sig['direction']} scalp closed at {hit_label}. +{pnl_pct:.1f}%",
                    data={"type": "signal_closed", "result": result, "ticker": ticker},
                )
            else:
                push._send_raw(
                    title=f"Scalp Stop Hit - {ticker}  {pnl_pct:.1f}%",
                    body=f"{sig['direction']} scalp stopped out. {pnl_pct:.1f}%",
                    data={"type": "signal_closed", "result": result, "ticker": ticker},
                )
        except Exception:
            pass

        # Remove from cache immediately so no duplicate close attempt
        _scalp_cache.pop(sig["ticker"], None)

        logger.info(
            f"[stream] SCALP CLOSED {sig['ticker']} {sig['direction']} "
            f"hit={hit.upper()} price={hit_price:.2f} pnl={pnl_pct:+.2f}%"
        )

    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[stream] Scalp close failed for {sig.get('id')}: {e}")


def _check_scalp_levels(symbol: str, bar_high: float, bar_low: float) -> None:
    """
    Check a 1-minute bar's high/low against all active scalp signals for this ticker.
    Called on every bar event — O(1) dict lookup, no DB query unless cache is stale.

    The bar's high and low represent the full price range touched during that minute,
    so they accurately reflect whether T1, T2, or SL was breached — even if the bar
    closed back inside the range (as happens with wicks).
    """
    global _scalp_cache, _scalp_cache_ts

    # Refresh cache if stale
    if time.monotonic() - _scalp_cache_ts > _SCALP_CACHE_TTL:
        _refresh_scalp_cache()

    sig = _scalp_cache.get(symbol)
    if not sig:
        return   # no active scalp signal for this ticker

    is_long   = sig["direction"] == "LONG"
    t1        = float(sig["target_one"])
    t2        = float(sig["target_two"])
    sl        = float(sig["stop_loss"])

    if is_long:
        # SL takes priority — if both high >= T1 and low <= SL, assume SL was hit
        # (conservative: protect capital first)
        if bar_low <= sl:
            _close_scalp_signal(sig, "sl", bar_low)
        elif bar_high >= t2:
            _close_scalp_signal(sig, "t2", bar_high)
        elif bar_high >= t1:
            _close_scalp_signal(sig, "t1", bar_high)
    else:  # SHORT
        if bar_high >= sl:
            _close_scalp_signal(sig, "sl", bar_high)
        elif bar_low <= t2:
            _close_scalp_signal(sig, "t2", bar_low)
        elif bar_low <= t1:
            _close_scalp_signal(sig, "t1", bar_low)


# ── Real-time level checker — ALL active signals ──────────────

def _refresh_rt_cache() -> None:
    """
    Load all active stock signals (every strategy) into the RT cache.
    Excludes scalping — those are already handled by _check_scalp_levels
    via bar high/low, which is more accurate than raw trade prices.
    Called at most once per _RT_CACHE_TTL seconds.
    """
    global _rt_cache, _rt_cache_ts
    try:
        key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
        from supabase import create_client as _sc
        sb = _sc(os.environ["SUPABASE_URL"], key)
        rows = (
            sb.table("signals")
            .select("id, ticker, direction, entry_price, stop_loss, target_one, target_two, strategy_type")
            .eq("status", "active")
            .neq("strategy_type", "scalping")   # scalping handled by bar checker
            .execute()
            .data
        ) or []

        new_cache: dict[str, list[dict]] = {}
        for r in rows:
            new_cache.setdefault(r["ticker"], []).append(r)

        _rt_cache    = new_cache
        _rt_cache_ts = time.monotonic()

        total = sum(len(v) for v in new_cache.values())
        if total:
            logger.debug(f"[stream] RT cache refreshed: {total} active signal(s) across {len(new_cache)} ticker(s)")
    except Exception as e:
        logger.debug(f"[stream] RT cache refresh failed: {e}")


def _close_rt_signal(sig: dict, hit: str, price: float) -> None:
    """
    Instantly close a signal when its T2 or stop-loss is breached on a live tick.
    Records accurate P&L and fires push notification.

    hit: "t2" | "sl"
    """
    global _rt_cache
    try:
        key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
        from supabase import create_client as _sc
        from engine import push as _push
        sb = _sc(os.environ["SUPABASE_URL"], key)

        entry    = float(sig["entry_price"])
        is_long  = sig["direction"] == "LONG"
        result   = "win" if hit == "t2" else "loss"
        hit_label = "Target 2" if hit == "t2" else "Stop Loss"

        pnl_pct = ((price - entry) / entry * 100) if is_long else ((entry - price) / entry * 100)
        pnl_abs = (price - entry) if is_long else (entry - price)

        sb.table("signals").update({
            "status":        "closed",
            "result":        result,
            "hit_target":    hit,
            "result_pct":    round(pnl_pct, 4),
            "result_pnl":    round(pnl_abs, 4),
            "closed_reason": "target_hit" if result == "win" else "stop_hit",
            "closed_at":     datetime.now(timezone.utc).isoformat(),
        }).eq("id", sig["id"]).execute()

        # Timeline event
        sign_str = "+" if pnl_pct > 0 else ""
        note = (
            f"{hit_label} hit @ ${price:.2f} — closed {sign_str}{pnl_pct:.1f}% "
            f"({'win' if result == 'win' else 'loss'})"
        )
        sb.table("signal_events").insert({
            "signal_id":  sig["id"],
            "event_type": "closed_win" if result == "win" else "closed_loss",
            "price":      price,
            "note":       note,
        }).execute()

        # Push notification
        ticker = sig["ticker"]
        try:
            if result == "win":
                _push._send_raw(
                    title=f"✅ T2 Hit — {ticker}  +{pnl_pct:.1f}%",
                    body=f"{sig['direction']} {(sig.get('strategy_type') or 'signal').replace('_',' ')} closed at full target.",
                    data={"type": "signal_closed", "result": "win", "ticker": ticker},
                )
            else:
                _push._send_raw(
                    title=f"🔴 Stop Hit — {ticker}  {pnl_pct:.1f}%",
                    body=f"{sig['direction']} stopped out. Position closed.",
                    data={"type": "signal_closed", "result": "loss", "ticker": ticker},
                )
        except Exception:
            pass

        # Evict from cache immediately — prevents duplicate close
        sigs = _rt_cache.get(ticker, [])
        _rt_cache[ticker] = [s for s in sigs if s["id"] != sig["id"]]

        logger.info(
            f"[stream] ⚡ RT CLOSE {ticker} {sig['direction']} "
            f"hit={hit.upper()} price={price:.2f} pnl={pnl_pct:+.2f}%"
        )

    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[stream] RT close failed for {sig.get('id')}: {e}")


def _handle_t1_rt(sig: dict, price: float) -> None:
    """
    T1 hit detected on a live trade tick (non-scalp signal).
    Does NOT close the signal — instead moves stop-loss to breakeven
    so any reversal is a scratch rather than a loss, then rides to T2.
    """
    try:
        key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
        from supabase import create_client as _sc
        from engine import push as _push
        sb = _sc(os.environ["SUPABASE_URL"], key)

        entry = float(sig["entry_price"])
        pct   = abs(price - entry) / entry * 100
        ticker = sig["ticker"]

        # Move SL to breakeven in DB
        sb.table("signals").update({"stop_loss": round(entry, 4)}).eq("id", sig["id"]).execute()

        # Timeline event
        sb.table("signal_events").insert({
            "signal_id":  sig["id"],
            "event_type": "t1_hit",
            "price":      price,
            "note":       f"🎯 T1 hit @ ${price:.2f} (+{pct:.1f}%) — stop moved to breakeven ${entry:.2f}",
        }).execute()

        # Push notification
        try:
            _push._send_raw(
                title=f"🎯 T1 Hit — {ticker}  +{pct:.1f}%",
                body=f"Stop moved to breakeven. Riding to T2. {sig['direction']} still open.",
                data={"type": "t1_breakeven", "ticker": ticker},
            )
        except Exception:
            pass

        # Update local cache so T1 doesn't re-trigger on next tick
        sig["stop_loss"] = entry

        logger.info(
            f"[stream] ⚡ RT T1 HIT {ticker} {sig['direction']} "
            f"price={price:.2f} pct=+{pct:.2f}% — SL moved to breakeven"
        )

    except Exception as e:
        logger.debug(f"[stream] RT T1 handler failed for {sig.get('id')}: {e}")


def _check_rt_levels(ticker: str, price: float) -> None:
    """
    Check a live trade price against all cached active signals for this ticker.
    Called at most once per second per ticker (throttled in on_trade).

    Decision tree per signal:
      T2 crossed  → close as win  (full target)
      T1 crossed  → move SL to breakeven, ride to T2 (don't close)
      SL crossed  → close as loss
      SL already at breakeven (T1 already hit) → only T2 / SL(=entry) matter
    """
    global _rt_cache, _rt_cache_ts

    # Refresh cache if stale
    if time.monotonic() - _rt_cache_ts > _RT_CACHE_TTL:
        _refresh_rt_cache()

    sigs = _rt_cache.get(ticker)
    if not sigs:
        return

    for sig in list(sigs):   # list() copy — we may mutate _rt_cache inside
        try:
            is_long = sig["direction"] == "LONG"
            t1      = float(sig["target_one"])
            t2      = float(sig["target_two"])
            sl      = float(sig["stop_loss"])
            entry   = float(sig["entry_price"])

            # Has T1 already been hit? (SL == entry means breakeven was set)
            t1_already_hit = abs(sl - entry) < 0.01

            if is_long:
                if price >= t2:
                    _close_rt_signal(sig, "t2", price)
                elif price >= t1 and not t1_already_hit:
                    _handle_t1_rt(sig, price)
                elif price <= sl:
                    _close_rt_signal(sig, "sl", price)
            else:   # SHORT
                if price <= t2:
                    _close_rt_signal(sig, "t2", price)
                elif price <= t1 and not t1_already_hit:
                    _handle_t1_rt(sig, price)
                elif price >= sl:
                    _close_rt_signal(sig, "sl", price)

        except Exception as e:
            logger.debug(f"[stream] RT level check error for {ticker}/{sig.get('id')}: {e}")


# ── Per-ticker scalp processor (unchanged) ────────────────────

def _process_bar_sync(symbol: str, close: float, volume: int) -> None:
    """
    Synchronous SMC scalping pipeline for a single ticker.
    Called from the scan executor when a 5-min bar close is detected.
    """
    try:
        session = _get_session()
        if session.get("blocked") or not session.get("market_open"):
            logger.debug(f"[stream] {symbol} scalp skipped — {session.get('block_reason', 'market closed')}")
            return

        regime = _get_regime()
        if regime.get("blocked"):
            logger.debug(f"[stream] {symbol} scalp skipped — {regime.get('block_reason', 'regime blocked')}")
            return

        logger.info(
            f"[stream] ⚡ Scalp bar: {symbol} @ {close:.2f} "
            f"vol={volume:,} | {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )

        from engine.runner import _process_smc_ticker, _supabase
        sb     = _supabase()
        config = {"type": "scalping", "interval": "5m", "period": "1d"}
        _process_smc_ticker(sb, symbol, config, regime=regime, session=session)

    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[stream] Scalp bar handler error for {symbol}: {e}", exc_info=True)


# ── Strategy boundary processor ───────────────────────────────

def _run_strategy_at_boundary(strategy_type: str) -> None:
    """
    Run a full strategy scan synchronously.
    Called from the scan executor at bar boundary events (15m or 1h close).
    """
    try:
        from engine.runner import run_strategy_by_type
        run_strategy_by_type(strategy_type)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"[stream] {strategy_type} boundary scan error: {e}", exc_info=True)


# ── Main stream coroutine ─────────────────────────────────────

async def run_stream() -> None:
    """
    Connect to Alpaca WebSocket and process bar events for all strategies.
    Subscribes to 1-minute bars for all watched tickers.
    Reconnects automatically with exponential backoff on any error.
    Runs indefinitely as a FastAPI background task.
    """
    api_key    = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_SECRET_KEY", "")

    if not api_key or not api_secret:
        logger.warning(
            "[stream] ALPACA_API_KEY / ALPACA_SECRET_KEY not set — "
            "real-time stream disabled. All strategies will be inactive."
        )
        return

    # Alpaca SDK requires a DataFeed enum — not a raw string
    from alpaca.data.enums import DataFeed
    # SIP = full real-time market data (requires Alpaca paid plan — confirmed active).
    # Override with ALPACA_DATA_FEED=iex only for free/paper-only accounts.
    feed_env  = os.environ.get("ALPACA_DATA_FEED", "sip").lower()
    feed      = DataFeed.SIP if feed_env == "sip" else DataFeed.IEX

    global _subscribed_tickers, _pending_tickers, _wss_ref, _on_trade_ref

    from engine.runner import ALL_TICKERS, SCALP_TICKERS
    scalp_set = set(SCALP_TICKERS)

    # Subscribe to all watched tickers — 1-min bars serve as clock ticks
    # for 5m (scalp), 15m (day_trade/flow), and 1h (swing) boundary detection.
    all_subscribe = list(dict.fromkeys(ALL_TICKERS))   # preserve order, deduplicate
    _subscribed_tickers = set(all_subscribe)           # track base set for dynamic subs

    logger.info(
        f"[stream] Starting event-driven stream — feed={feed.value.upper()} | "
        f"{len(all_subscribe)} tickers subscribed | "
        f"strategies: scalping(5m) day_trade/options_flow/dark_pool(15m) swing_trade(1h)"
    )

    # ── Startup grace period (Railway / Fly.io rolling deploys) ──────────
    # Both Railway and Fly.io start the new container before the old one stops.
    # Alpaca allows only 1 concurrent WebSocket per account, so both
    # containers briefly fight for the same connection → "connection limit
    # exceeded" spam.  Waiting here gives the old container time to die
    # and release its connection before we try to connect.
    # Set STREAM_STARTUP_DELAY_S=0 to disable if running multiple workers.
    _on_railway = bool(os.environ.get("RAILWAY_PROJECT_ID") or os.environ.get("RAILWAY_ENVIRONMENT"))
    _on_fly     = bool(os.environ.get("FLY_APP_NAME") or os.environ.get("FLY_MACHINE_ID"))
    # 60 s gives Fly.io's 30 s kill_timeout + ~30 s for Alpaca to release
    # the slot after the old instance closes the socket.
    _startup_delay = int(os.environ.get("STREAM_STARTUP_DELAY_S", "60" if (_on_railway or _on_fly) else "0"))
    if _startup_delay > 0:
        logger.info(
            f"[stream] Startup grace period — waiting {_startup_delay}s for "
            f"previous deployment to release Alpaca connection..."
        )
        await asyncio.sleep(_startup_delay)

    # Pre-warm RT signal cache so first trades don't miss any active signals
    try:
        _refresh_rt_cache()
        _refresh_scalp_cache()
    except Exception as _e:
        logger.debug(f"[stream] Cache pre-warm failed: {_e}")

    reconnect_delay = 5

    while True:
        try:
            from alpaca.data.live import StockDataStream

            wss = StockDataStream(api_key, api_secret, feed=feed)

            async def on_bar(bar) -> None:
                global _last_15m_barrier, _last_1h_barrier

                symbol   = bar.symbol
                close    = float(bar.close)
                bar_high = float(bar.high)
                bar_low  = float(bar.low)
                volume   = int(bar.volume)

                # Parse bar timestamp into ET for boundary detection
                try:
                    ts_et = bar.timestamp.astimezone(ET)
                except Exception:
                    ts_et = datetime.now(ET)

                minute  = ts_et.minute
                hour    = ts_et.hour
                min_key = hour * 60 + minute   # unique key per minute-of-day (0-1439)

                # ── EVERY bar: check scalp T1/SL in real-time ─────────────────
                # Uses bar high/low (wicks) so we catch levels touched intra-bar.
                # Runs before any boundary logic so closes fire as fast as possible.
                _scan_executor.submit(_check_scalp_levels, symbol, bar_high, bar_low)

                # ── Scalping: every 5-min bar close, per ticker ───────────────
                # Fires for each SCALP ticker individually as its bar arrives.
                # This gives sub-5-second latency per ticker (vs polling which
                # would fire all tickers at once on a fixed schedule).
                if symbol in scalp_set and minute % 5 == 0:
                    _scan_executor.submit(
                        _process_bar_sync, symbol, close, volume
                    )

                # ── 15-min bar close: day_trade + options_flow + dark_pool ────
                # Deduplication: only the FIRST ticker's bar at each 15-min
                # boundary fires the scan — subsequent bars that minute are ignored.
                if minute % 15 == 0 and min_key != _last_15m_barrier:
                    _last_15m_barrier = min_key
                    logger.info(
                        f"[stream] ⏱ 15-min bar close @ "
                        f"{ts_et.strftime('%H:%M ET')} — "
                        f"firing day_trade / options_flow / dark_pool"
                    )
                    _scan_executor.submit(_run_strategy_at_boundary, "day_trade")
                    _scan_executor.submit(_run_strategy_at_boundary, "options_flow")
                    _scan_executor.submit(_run_strategy_at_boundary, "dark_pool")

                # ── 1-hour bar close: swing_trade ─────────────────────────────
                if minute == 0 and min_key != _last_1h_barrier:
                    _last_1h_barrier = min_key
                    logger.info(
                        f"[stream] ⏱ 1-hour bar close @ "
                        f"{ts_et.strftime('%H:%M ET')} — firing swing_trade"
                    )
                    _scan_executor.submit(_run_strategy_at_boundary, "swing_trade")

            # ── Trade handler: price broadcast + real-time level checks ──────
            async def on_trade(trade) -> None:
                ticker = trade.symbol
                price  = float(trade.price)

                # 1. Feed price to WebSocket clients (always — no throttle)
                try:
                    from engine.price_store import update as price_update
                    price_update(ticker, price)
                except Exception:
                    pass   # never let price broadcast errors kill the stream

                # 2. Real-time T1/T2/SL check for ALL active non-scalp signals.
                #    Throttled to at most once per second per ticker so we don't
                #    flood the executor with thousands of tasks on liquid stocks.
                now = time.monotonic()
                if now - _rt_last_check.get(ticker, 0.0) >= _RT_THROTTLE_S:
                    _rt_last_check[ticker] = now
                    _scan_executor.submit(_check_rt_levels, ticker, price)

                # 3. Tick-triggered scalp scan — fire the full SMC pipeline NOW
                #    instead of waiting for the next 5-min bar close.
                #    If a setup forms at minute 0:30 of a bar, this fires within
                #    1 second; the bar-close path would wait up to 4.5 more min.
                #    Throttled to _TICK_SCALP_THROTTLE_S (5 min) per ticker so
                #    a single scalp scan can't be queued multiple times.
                if ticker in scalp_set:
                    if now - _tick_scalp_last.get(ticker, 0.0) >= _TICK_SCALP_THROTTLE_S:
                        _tick_scalp_last[ticker] = now
                        _scan_executor.submit(_process_bar_sync, ticker, price, 0)

            wss.subscribe_bars(on_bar, *all_subscribe)
            wss.subscribe_trades(on_trade, *all_subscribe)

            # ── Register live references for dynamic ticker subscriptions ──
            _wss_ref      = wss
            _on_trade_ref = on_trade

            # Apply any tickers that were requested by WS clients before
            # this connection was established (or queued during a reconnect).
            loop = asyncio.get_running_loop()
            if _pending_tickers:
                extra = list(_pending_tickers - set(all_subscribe))
                if extra:
                    try:
                        await loop.run_in_executor(
                            None, wss.subscribe_trades, on_trade, *extra
                        )
                        logger.info(
                            f"[stream] ✅ Applied {len(extra)} pending dynamic ticker(s): {extra}"
                        )
                    except Exception as _pe:
                        logger.warning(f"[stream] Pending ticker subscribe failed: {_pe}")
                _pending_tickers.clear()

            logger.info(
                f"[stream] ✅ Connected to Alpaca {feed.value.upper()} — "
                f"bars + trades subscribed ({len(all_subscribe)} tickers + "
                f"{len(_subscribed_tickers) - len(all_subscribe)} dynamic)"
            )
            reconnect_delay = 5   # reset on successful connect

            await loop.run_in_executor(None, wss.run)

            # Stream ended — clear live references before reconnecting.
            # Re-queue any dynamic tickers so they're reapplied on next connect.
            _wss_ref      = None
            _on_trade_ref = None
            dynamic = _subscribed_tickers - set(all_subscribe)
            if dynamic:
                _pending_tickers.update(dynamic)

            logger.warning("[stream] Stream ended cleanly — reconnecting in 5s")
            await asyncio.sleep(5)

        except asyncio.CancelledError:
            # Stop the Alpaca WebSocket BEFORE clearing the ref so the old
            # instance releases the SIP connection before the new deployment
            # tries to claim it.  Without this, Fly.io rolling deploys leave
            # a zombie connection and the new instance gets "connection limit
            # exceeded" for up to 60 s.
            try:
                wss.stop()
                logger.info("[stream] Alpaca WebSocket stopped cleanly on shutdown")
            except Exception as _se:
                logger.debug(f"[stream] wss.stop() error (non-fatal): {_se}")
            _wss_ref      = None
            _on_trade_ref = None
            logger.info("[stream] Stream task cancelled — shutting down")
            _scan_executor.shutdown(wait=False)
            return

        except Exception as e:
            _wss_ref      = None
            _on_trade_ref = None
            sentry_sdk.capture_exception(e)
            err_str = str(e).lower()
            # Treat connection limit, 429, AND TimeoutError the same way:
            # back off for 60 s minimum.
            #
            # TimeoutError root cause: Fly.io's default kill_timeout is 5 s.
            # If the old instance is SIGKILL'd before our graceful shutdown
            # sends a proper FIN to Alpaca, Alpaca holds the dead TCP slot
            # for up to 120 s via keepalive.  The new instance gets
            # TimeoutError (not 429) because Alpaca accepts the TCP handshake
            # but never completes the WebSocket upgrade.
            # kill_timeout = "30s" in fly.toml fixes the root cause; this
            # 60 s backoff is the safety net for any remaining races.
            _is_conn_limit = (
                "connection limit" in err_str
                or "429" in err_str
                or isinstance(e, (TimeoutError, asyncio.TimeoutError))
            )
            if _is_conn_limit:
                wait = max(reconnect_delay, 60)
                logger.warning(
                    f"[stream] Alpaca connection unavailable ({type(e).__name__}) — "
                    f"backing off {wait}s before retry"
                )
                await asyncio.sleep(wait)
                reconnect_delay = min(wait * 2, 120)
            else:
                logger.error(
                    f"[stream] Connection error: {e} — "
                    f"reconnecting in {reconnect_delay}s"
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 120)
