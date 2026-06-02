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

# ── RT level executor ──────────────────────────────────────────
# Dedicated pool for T1/T2/SL level checks ONLY — completely separate
# from _scan_executor so a full strategy scan (which can take 2-10 s)
# never delays a level check by even a single tick.
#
# max_workers=2: one thread handles the current check while the second
# absorbs the next tick that arrives during it. More than 2 would be
# wasteful — level checks are fast (<100 ms) and we throttle to 1/s.
_rt_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="sb_rt",
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

# ── Tick-triggered day_trade scanner ─────────────────────────────────────────
# Fire a full day_trade SMC scan the moment a ticker shows ≥0.4% momentum in
# the last 60 seconds — without waiting for the next 15-min bar boundary.
# This means new day_trade signals can appear within seconds of a setup forming,
# not at the next fixed 15-min clock tick.
#
# Throttle: 900 s (15 min) per ticker — same cadence as bar-close day_trade scans.
# Gate: only fires when _has_tick_momentum() confirms the move is real.
_TICK_DAYTRADE_THROTTLE_S: float = 900.0
_tick_daytrade_last: dict[str, float] = {}   # ticker → last tick-scan monotonic time

# ── Price momentum buffer ─────────────────────────────────────────────────────
# A rolling 60-second window of trade prices per ticker.
# Used by _has_tick_momentum() to detect fast directional moves before the next
# bar close — the trigger gate for tick-triggered day_trade scans.
#
# Structure: { ticker: [(price, monotonic_ts), ...] }
# Pruned on every write so memory stays bounded.
_PRICE_BUFFER_WINDOW_S: float = 600.0   # 10-min rolling window — enough for structural analysis
_MOMENTUM_THRESHOLD:    float = 0.004   # 0.4% move in 60 s → day_trade scan fires
_price_buffer: dict[str, list] = {}     # ticker → [(price, monotonic_ts), ...]

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
_on_bar_ref                   = None    # stored bar handler (dynamic bar subs)

# ── Compression breakout watch (per-tick firing) ──────────────────────────────
# Staged on bar close by runner._process_predictive_ticker; checked on EVERY
# tick in on_trade so a breakout fires the instant price crosses the envelope
# edge — not at the next 15m scan. {ticker: (range_high, range_low, atr, staged_ts)}
import threading as _threading
_compression_zones: dict[str, tuple] = {}
_compression_lock = _threading.Lock()
_compression_fired: dict[str, float] = {}      # ticker -> last fire ts (re-fire throttle)
_COMPRESSION_ZONE_TTL   = 2 * 3600             # drop a staged zone after 2h
_COMPRESSION_FIRE_THROTTLE = 1800              # don't re-fire same ticker within 30 min
_COMPRESSION_BREAKOUT_PCT  = 0.10              # price must clear edge by this % (anti-wick)

# Breakout-quality: the confirming 1m bar must close in the top fraction of its
# range for a LONG break (bottom fraction for SHORT). Rejects "rejection" bars —
# a red candle with a big upper wick that pokes the level then closes weak
# (the MARA bull-trap pattern, 2026-05-28). Fail-open if the bar range is unknown.
_STRONG_CLOSE_FRAC = 0.60


def _is_strong_close(direction: str, close: float,
                     bar_high: float | None, bar_low: float | None) -> bool:
    if not bar_high or not bar_low or bar_high <= bar_low:
        return True
    pos = (close - bar_low) / (bar_high - bar_low)   # 0 = at low, 1 = at high
    return pos >= _STRONG_CLOSE_FRAC if direction == "LONG" else pos <= (1.0 - _STRONG_CLOSE_FRAC)


# ── Volume-surge confirmation ─────────────────────────────────────────────────
# The slow-bleed losers (NKE/MARA/SNOW/NOW, 2026-05-28) broke a level with no
# participation and just faded. Require the confirming 1m bar's volume to beat a
# trailing average — real breakouts have buyers. Fail-open until we have history.
_vol_buffer: dict[str, list] = {}
_VOL_SURGE_MULT  = 1.5
_VOL_MIN_HISTORY = 6


def _record_volume(ticker: str, vol) -> None:
    if not vol or vol <= 0:
        return
    b = _vol_buffer.setdefault(ticker, [])
    b.append(float(vol))
    if len(b) > 20:
        del b[0]


def _vol_surge(ticker: str, vol) -> bool:
    b = _vol_buffer.get(ticker, [])
    if len(b) < _VOL_MIN_HISTORY or not vol:
        return True                                  # not enough history → allow
    avg = sum(b) / len(b)
    return avg <= 0 or vol >= _VOL_SURGE_MULT * avg


# ── Market-regime alignment ───────────────────────────────────────────────────
# 6 of 7 predictive losers were LONG, fired into a weak tape. The predictive path
# had no market awareness. Block LONGs in a bearish regime / SHORTs in a bullish
# one (coarse SPY/VIX/200MA regime, already cached — zero extra cost).
def _market_allows(direction: str) -> bool:
    try:
        rt = (_get_regime() or {}).get("regime_type", "RANGING")
    except Exception:
        return True
    if direction == "LONG"  and rt in ("TRENDING_BEAR", "RISK_OFF", "PANIC"):
        return False
    if direction == "SHORT" and rt == "TRENDING_BULL":
        return False
    return True


# ── Breakout RETEST entry ─────────────────────────────────────────────────────
# Don't chase the breakout candle (BKNG bought the spike, faded to a 0.2% stop,
# 2026-05-28). On a confirmed swing/compression break, arm a "retest pending":
# wait for price to pull back to the broken level and HOLD, then enter near the
# level (better price + room for an ATR stop below it). Cancel if price breaks
# back through the level (fake-out) or no retest within the window (don't chase).
# {ticker: {"direction","level","atr","detector","break_ts"}}
_retest_pending: dict[str, dict] = {}
_retest_lock = _threading.Lock()
_RETEST_WINDOW_SEC = 45 * 60     # give up waiting for a retest after 45 min
_RETEST_BAND_PCT   = 0.25        # pullback must come within this % of the level


def _arm_retest(ticker: str, direction: str, level: float, atr: float, detector: str) -> None:
    import time as _t
    with _retest_lock:
        _retest_pending[ticker] = {"direction": direction, "level": float(level),
                                   "atr": float(atr), "detector": detector, "break_ts": _t.time()}
    logger.info(f"[stream] {ticker} {detector} broke {direction} @ {level:.2f} — awaiting retest")


def clear_all_zones() -> None:
    """Clear all armed per-tick zones — called by the overnight scheduler
    (~12:30 AM ET) so zones survive after-hours for admin analysis and re-arm
    fresh next session."""
    with _compression_lock:  _compression_zones.clear()
    with _pullback_lock:     _pullback_zones.clear()
    with _swing_lock:        _swing_zones.clear()
    with _zone_relaxed_lock: _zone_relaxed.clear()
    with _retest_lock:       _retest_pending.clear()
    _persist_zones(force=True)
    logger.info("[stream] Cleared all armed zones (overnight)")


def _check_retest(ticker: str, close: float,
                  bar_high: float | None = None, bar_low: float | None = None) -> None:
    """On 1m close: fire a breakout once price retests the broken level and holds."""
    import time as _t
    p = _retest_pending.get(ticker)
    if p is None:
        return
    now = _t.time()
    if now - p["break_ts"] > _RETEST_WINDOW_SEC:
        with _retest_lock:
            _retest_pending.pop(ticker, None)        # ran away / never retested — skip
        return

    level, direction, atr, detector = p["level"], p["direction"], p["atr"], p["detector"]
    band = level * (_RETEST_BAND_PCT / 100)

    if direction == "LONG":
        broke_back = close < level                    # lost the level → fake-out
        retested   = (bar_low is not None and bar_low <= level + band) and close >= level
    else:  # SHORT
        broke_back = close > level
        retested   = (bar_high is not None and bar_high >= level - band) and close <= level

    if broke_back:
        with _retest_lock:
            _retest_pending.pop(ticker, None)
        logger.info(f"[stream] {ticker} {detector} retest FAILED (lost {level:.2f}) — cancelled")
        return
    if not retested:
        return

    with _retest_lock:
        _retest_pending.pop(ticker, None)
    try:
        from engine import runner as _runner
        fire = {"SWING_BREAKOUT": _runner.fire_swing_breakout,
                "COMPRESSION":    _runner.fire_compression_breakout,
                "PULLBACK":       _runner.fire_pullback_reclaim}.get(detector)
        if fire:
            _scan_executor.submit(fire, ticker, direction, close, None, level)
    except Exception as e:
        logger.debug(f"[stream] retest fire dispatch failed for {ticker}: {e}")


def stage_compression_zone(ticker: str, range_high: float, range_low: float, atr: float) -> bool:
    """Register a compression envelope for per-tick breakout watching.
    Returns True if this is a FRESH arm (ticker wasn't already staged), so the
    caller can log the arming once; re-stages preserve the original arm time."""
    import time as _t
    with _compression_lock:
        prev = _compression_zones.get(ticker)
        armed_ts = prev[3] if prev else _t.time()   # preserve original arm time
        _compression_zones[ticker] = (range_high, range_low, atr, armed_ts)
        return prev is None


def clear_compression_zone(ticker: str) -> None:
    """Remove a ticker from the compression watch (no longer compressed)."""
    with _compression_lock:
        _compression_zones.pop(ticker, None)


# ── Pullback reclaim watch (per-tick firing) ──────────────────────────────────
# Staged on bar close; fired the instant price crosses the reclaim level.
# {ticker: (direction, reclaim_level, stop_ref, atr, staged_ts)}
_pullback_zones: dict[str, tuple] = {}
_pullback_lock = _threading.Lock()
_pullback_fired: dict[str, float] = {}


def stage_pullback_zone(ticker: str, direction: str, reclaim_level: float,
                        stop_ref: float, atr: float) -> bool:
    """Register a pullback reclaim level for per-tick watching. Returns True on
    a fresh arm (preserves original arm time across re-stages)."""
    import time as _t
    with _pullback_lock:
        prev = _pullback_zones.get(ticker)
        armed_ts = prev[4] if prev else _t.time()
        _pullback_zones[ticker] = (direction, reclaim_level, stop_ref, atr, armed_ts)
        return prev is None


def clear_pullback_zone(ticker: str) -> None:
    with _pullback_lock:
        _pullback_zones.pop(ticker, None)


def _check_pullback_reclaim(ticker: str, price: float,
                            bar_high: float | None = None, bar_low: float | None = None,
                            bar_vol: float | None = None) -> None:
    """Called on 1m close. Fire when the bar closes across the reclaim level."""
    import time as _t
    zone = _pullback_zones.get(ticker)
    if zone is None:
        return
    direction, reclaim_level, stop_ref, atr, staged_ts = zone
    now = _t.time()
    if now - staged_ts > _COMPRESSION_ZONE_TTL:          # reuse 2h TTL
        with _pullback_lock:
            _pullback_zones.pop(ticker, None)
        return
    if now - _pullback_fired.get(ticker, 0) < _COMPRESSION_FIRE_THROTTLE:
        return

    crossed = (direction == "LONG"  and price >= reclaim_level) or \
              (direction == "SHORT" and price <= reclaim_level)
    if not crossed:
        return
    if not _is_strong_close(direction, price, bar_high, bar_low):
        return   # rejection bar — weak close, skip
    if not _vol_surge(ticker, bar_vol):
        return   # no participation — low-volume reclaim tends to fade
    if not _market_allows(direction):
        return   # fighting the market regime

    level = reclaim_level
    with _pullback_lock:
        _pullback_fired[ticker] = now
        _pullback_zones.pop(ticker, None)
    # Don't fire on the raw reclaim (INTC/UMAC fired into reversals) — wait for a
    # retest/hold of the reclaim level, with the stop placed beyond it.
    _arm_retest(ticker, direction, level, atr, "PULLBACK")


# ── Swing-high breakout watch (per-tick firing) ───────────────────────────────
# Staged on bar close; fired when price crosses the recent swing high/low.
# {ticker: (swing_high, swing_low, atr, staged_ts)}  (0.0 = side not watched)
_swing_zones: dict[str, tuple] = {}
_swing_lock = _threading.Lock()
_swing_fired: dict[str, float] = {}
_SWING_BREAKOUT_PCT = 0.05   # price must clear the level by this % (anti-wick)


def stage_swing_zone(ticker: str, swing_high: float, swing_low: float, atr: float) -> bool:
    """Returns True on a fresh arm (preserves original arm time across re-stages)."""
    import time as _t
    with _swing_lock:
        prev = _swing_zones.get(ticker)
        armed_ts = prev[3] if prev else _t.time()
        _swing_zones[ticker] = (swing_high, swing_low, atr, armed_ts)
        return prev is None


def clear_swing_zone(ticker: str) -> None:
    with _swing_lock:
        _swing_zones.pop(ticker, None)


# ── Zone persistence across worker restarts ───────────────────────────────────
# Staged zones live in worker memory; every deploy/restart wiped them, so the
# per-tick detectors reset to zero (the "no signals all day" bug 2026-05-28).
# Persist the three zone dicts to DURABLE Postgres (engine_kv) on stage and
# reload on startup so zones survive restarts. Postgres replaced Redis here
# (2026-05-28): Redis snapshot writes were timing out ('Timeout reading from
# socket'), making restore + the admin display unreliable. Per-tick reads stay
# in-memory (fast); Postgres is only touched on stage (throttled) + startup.
_ZONE_KV_KEY    = "stream:zones:v1"
_ZONE_PERSIST_MIN_INTERVAL_S = 15.0   # throttle writes (scan calls this ~40×/scan)
_last_zone_persist_ts = 0.0

# Per-ticker "relaxed-eligible" state (computed at staging): is this ticker
# currently extended past the standard cap with trend+volume confirming, so a
# fire would use the wider momentum cap? Shown as a badge in the Armed Zones UI.
_zone_relaxed: dict[str, dict] = {}
_zone_relaxed_lock = _threading.Lock()


def set_zone_relaxed(ticker: str, state: dict | None) -> None:
    """Record (or clear) the relaxed-eligible state for a ticker."""
    with _zone_relaxed_lock:
        if state and state.get("eligible"):
            _zone_relaxed[ticker] = {"ext_atr": state.get("ext_atr"),
                                     "direction": state.get("direction")}
        else:
            _zone_relaxed.pop(ticker, None)


def _zone_supabase():
    """Service-role Supabase client for zone persistence (worker context)."""
    import os
    from supabase import create_client as _sc
    key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
    return _sc(os.environ["SUPABASE_URL"], key)


def _persist_zones(force: bool = False) -> None:
    """Snapshot all three zone dicts to Postgres engine_kv (single-row upsert).

    Throttled to one write per _ZONE_PERSIST_MIN_INTERVAL_S so a 40-ticker scan
    doesn't hammer the DB. The in-memory dicts are always current; the durable
    snapshot lags by at most the throttle interval — that only affects
    cross-restart restore + the admin armed-zone display. Single-row JSONB
    upsert is atomic, so readers never see a half-written snapshot.
    """
    global _last_zone_persist_ts
    import time as _t
    from datetime import datetime, timezone
    now = _t.time()
    if not force and (now - _last_zone_persist_ts) < _ZONE_PERSIST_MIN_INTERVAL_S:
        return
    core = {
        "compression": {k: list(v) for k, v in _compression_zones.items()},
        "pullback":    {k: list(v) for k, v in _pullback_zones.items()},
        "swing":       {k: list(v) for k, v in _swing_zones.items()},
    }
    try:
        snap = {**core, "relaxed": dict(_zone_relaxed)}
        try:
            _zone_supabase().table("engine_kv").upsert({
                "key": _ZONE_KV_KEY, "value": snap,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
        except Exception as _re:
            # Never let the optional `relaxed` payload (e.g. a non-JSON value)
            # block the core zone snapshot — retry without it so the armed
            # display + restore stay current.
            logger.debug(f"[stream] zone persist w/ relaxed failed ({_re}); retrying core-only")
            _zone_supabase().table("engine_kv").upsert({
                "key": _ZONE_KV_KEY, "value": {**core, "relaxed": {}},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
        _last_zone_persist_ts = now   # only advance on success so failures retry
    except Exception as e:
        logger.debug(f"[stream] zone persist failed: {e}")


def load_zones_from_db() -> set[str]:
    """Restore staged zones from Postgres on worker startup (survives restarts).

    Returns the set of zone tickers so the caller can re-subscribe them to the
    live stream — otherwise restored zones sit in memory but receive no bars
    until the next scan re-arms them.
    """
    try:
        rows = (_zone_supabase().table("engine_kv")
                .select("value").eq("key", _ZONE_KV_KEY).limit(1).execute().data) or []
        snap = rows[0]["value"] if rows else None
        if not snap:
            return set()
        with _compression_lock:
            _compression_zones.update({k: tuple(v) for k, v in (snap.get("compression") or {}).items()})
        with _pullback_lock:
            _pullback_zones.update({k: tuple(v) for k, v in (snap.get("pullback") or {}).items()})
        with _swing_lock:
            _swing_zones.update({k: tuple(v) for k, v in (snap.get("swing") or {}).items()})
        with _zone_relaxed_lock:
            _zone_relaxed.update(snap.get("relaxed") or {})
        logger.info(f"[stream] Restored zones from DB — "
                    f"comp={len(_compression_zones)} pb={len(_pullback_zones)} swing={len(_swing_zones)}")
        return set(_compression_zones) | set(_pullback_zones) | set(_swing_zones)
    except Exception as e:
        logger.debug(f"[stream] zone restore failed: {e}")
        return set()


def _check_swing_breakout(ticker: str, price: float,
                          bar_high: float | None = None, bar_low: float | None = None,
                          bar_vol: float | None = None) -> None:
    """Called on 1m close. Fire when the bar closes beyond the staged swing high/low."""
    import time as _t
    zone = _swing_zones.get(ticker)
    if zone is None:
        return
    swing_high, swing_low, atr, staged_ts = zone
    now = _t.time()
    if now - staged_ts > _COMPRESSION_ZONE_TTL:
        with _swing_lock:
            _swing_zones.pop(ticker, None)
        return
    if now - _swing_fired.get(ticker, 0) < _COMPRESSION_FIRE_THROTTLE:
        return

    direction = None
    if swing_high > 0 and price >= swing_high * (1 + _SWING_BREAKOUT_PCT / 100):
        direction = "LONG"
    elif swing_low > 0 and price <= swing_low * (1 - _SWING_BREAKOUT_PCT / 100):
        direction = "SHORT"
    if direction is None:
        return
    if not _is_strong_close(direction, price, bar_high, bar_low):
        return   # rejection bar — weak close, skip (MARA bull-trap guard)
    if not _vol_surge(ticker, bar_vol):
        return   # no participation — low-volume break tends to fade
    if not _market_allows(direction):
        return   # fighting the market regime

    level = swing_high if direction == "LONG" else swing_low
    with _swing_lock:
        _swing_fired[ticker] = now
        _swing_zones.pop(ticker, None)
    # Don't fire at the spike — wait for a retest of the broken level.
    _arm_retest(ticker, direction, level, atr, "SWING_BREAKOUT")


def _check_compression_breakout(ticker: str, price: float,
                                bar_high: float | None = None, bar_low: float | None = None,
                                bar_vol: float | None = None) -> None:
    """
    Called on 1m close. If `ticker` has a staged compression zone and the bar
    closed beyond an edge (with a strong close), fire a breakout signal via
    runner (in the scan executor so we don't block the event loop).
    """
    import time as _t
    zone = _compression_zones.get(ticker)
    if zone is None:
        return
    range_high, range_low, atr, staged_ts = zone
    now = _t.time()

    # Expire stale zones
    if now - staged_ts > _COMPRESSION_ZONE_TTL:
        with _compression_lock:
            _compression_zones.pop(ticker, None)
        return

    # Re-fire throttle
    if now - _compression_fired.get(ticker, 0) < _COMPRESSION_FIRE_THROTTLE:
        return

    upper = range_high * (1 + _COMPRESSION_BREAKOUT_PCT / 100)
    lower = range_low  * (1 - _COMPRESSION_BREAKOUT_PCT / 100)

    direction = None
    if price >= upper:
        direction = "LONG"
    elif price <= lower:
        direction = "SHORT"
    if direction is None:
        return
    if not _is_strong_close(direction, price, bar_high, bar_low):
        return   # rejection bar — weak close, skip
    if not _vol_surge(ticker, bar_vol):
        return   # no participation — low-volume break tends to fade
    if not _market_allows(direction):
        return   # fighting the market regime

    level = range_high if direction == "LONG" else range_low
    with _compression_lock:
        _compression_fired[ticker] = now
        _compression_zones.pop(ticker, None)
    # Don't fire at the spike — wait for a retest of the broken edge.
    _arm_retest(ticker, direction, level, atr, "COMPRESSION")


async def subscribe_extra_tickers(tickers: list[str]) -> None:
    """
    Dynamically subscribe additional tickers to the live Alpaca trade stream.

    Called from /ws/prices when a client subscribes to tickers not in ALL_TICKERS
    (e.g. custom watchlist symbols). Once subscribed, Alpaca starts pushing trade
    ticks for those symbols → price_store.update() → broadcast to WS clients.

    Safe to call at any time — if the stream is not yet connected, the tickers are
    queued and applied as soon as the stream comes up (or on the next reconnect).
    """
    global _subscribed_tickers, _pending_tickers, _wss_ref, _on_trade_ref, _on_bar_ref

    new = [t for t in tickers if t not in _subscribed_tickers]
    if not new:
        return

    _subscribed_tickers.update(new)

    if _wss_ref is None or _on_trade_ref is None or _on_bar_ref is None:
        _pending_tickers.update(new)
        logger.debug(f"[stream] Queued dynamic tickers (stream not ready yet): {new}")
        return

    # subscribe_trades()/subscribe_bars() call asyncio.run_coroutine_threadsafe()
    # internally — blocks the calling thread, not the FastAPI event loop. Run in
    # an executor so we don't block the async event loop.
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _wss_ref.subscribe_trades, _on_trade_ref, *new)
        # CRITICAL: also subscribe BARS. Breakout confirmation runs on the 1m bar
        # close (on_bar); without a bar subscription, dynamically-added movers
        # would get trades but no bars → their armed zones could never confirm.
        await loop.run_in_executor(None, _wss_ref.subscribe_bars, _on_bar_ref, *new)
        logger.info(f"[stream] ✅ Dynamic trade+bar subscription added: {new}")
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
        except Exception as _ev_e:
            # signal_events is a timeline log — if insert fails the actual
            # signal close still happened. Surface as warning so persistent
            # schema/Supabase issues become visible.
            logger.warning(f"[stream] signal_events insert failed for {sig['id']}: {_ev_e}")

        # Push notification — failure here = user doesn't see their win/loss
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
        except Exception as _push_e:
            logger.warning(f"[stream] scalp close push failed for {sig['id']}: {_push_e}")

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
            .select("id, ticker, direction, entry_price, stop_loss, target_one, target_two, strategy_type, created_at, score_breakdown")
            .eq("status", "active")
            .neq("strategy_type", "scalping")   # scalping handled by bar checker
            .execute()
            .data
        ) or []

        new_cache: dict[str, list[dict]] = {}
        for r in rows:
            # TREND_MOMENTUM exits on DAILY closes via momentum_monitor — never
            # on an intraday tick. Keep it out of the real-time SL/TP path so a
            # wick can't close it.
            if ((r.get("score_breakdown") or {}).get("detector_source")) == "TREND_MOMENTUM":
                continue
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
                    data={"type": "signal_closed", "result": "win", "ticker": ticker,
                          "signal_id": str(sig["id"])},
                )
            else:
                _push._send_raw(
                    title=f"🔴 Stop Hit — {ticker}  {pnl_pct:.1f}%",
                    body=f"{sig['direction']} stopped out. Position closed.",
                    data={"type": "signal_closed", "result": "loss", "ticker": ticker,
                          "signal_id": str(sig["id"])},
                )
        except Exception as _push_e:
            logger.warning(f"[stream] RT close push failed for {sig.get('id')}: {_push_e}")

        # Evict from cache immediately — prevents duplicate close
        sigs = _rt_cache.get(ticker, [])
        _rt_cache[ticker] = [s for s in sigs if s["id"] != sig["id"]]

        # Clear advisor state — signal is closed, no more advice needed
        try:
            from engine import signal_advisor as _advisor
            _advisor.evict(sig["id"])
        except Exception:
            pass

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

    Decision — engine uses its own analysis to decide close vs trail:

      scalping   → already closed by _check_scalp_levels, never reaches here
      day_trade  → regime-aware:
                     TRENDING_BULL / TRENDING_BEAR  → trail to T2
                       (move SL to breakeven, ride the momentum)
                     RANGING / HIGH_VOL / PANIC / LOW_VOL → close at T1
                       (take profit, structure unlikely to extend cleanly)
      swing_trade → always trail to T2
                       (designed for bigger moves; T1 is a milestone not an exit)

    In all close cases: DB updated, card disappears, "Book Profit" push fires.
    In all trail cases: SL moved to breakeven, "T1 Hit — Riding to T2" push fires.
    """
    try:
        key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
        from supabase import create_client as _sc
        from engine import push as _push
        sb = _sc(os.environ["SUPABASE_URL"], key)

        entry    = float(sig["entry_price"])
        is_long  = sig["direction"] == "LONG"
        ticker   = sig["ticker"]
        strategy = sig.get("strategy_type", "day_trade")

        pnl_pct = ((price - entry) / entry * 100) if is_long else ((entry - price) / entry * 100)
        pnl_abs = (price - entry) if is_long else (entry - price)

        # ── Decide: close at T1 or trail to T2? ──────────────────────────────
        # Swing always trails. Scalping never reaches here (handled by bar checker).
        # Day trade defers to the cached regime — no extra API call needed.
        should_trail = False
        if strategy == "swing_trade":
            should_trail = True
        elif strategy == "day_trade":
            try:
                regime_type = _get_regime().get("regime_type", "RANGING")
                should_trail = regime_type in ("TRENDING_BULL", "TRENDING_BEAR")
            except Exception:
                should_trail = False   # safe default: book the profit

        if should_trail:
            # ── TRAIL: move SL to breakeven, keep riding to T2 ───────────────
            sb.table("signals").update(
                {"stop_loss": round(entry, 4)}
            ).eq("id", sig["id"]).execute()

            sb.table("signal_events").insert({
                "signal_id":  sig["id"],
                "event_type": "t1_hit",
                "price":      price,
                "note":       (
                    f"🎯 T1 hit @ ${price:.2f} (+{pnl_pct:.1f}%) — "
                    f"stop moved to breakeven ${entry:.2f}, riding to T2"
                ),
            }).execute()

            try:
                _push._send_raw(
                    title=f"🎯 T1 Hit — {ticker}  +{pnl_pct:.1f}%",
                    body=f"Trending regime — stop at breakeven. Riding to T2. {sig['direction']} open.",
                    data={"type": "t1_breakeven", "ticker": ticker,
                          "signal_id": str(sig["id"])},
                )
            except Exception:
                pass

            # Update local cache SL so T1 doesn't re-trigger on next tick
            sig["stop_loss"] = entry

            logger.info(
                f"[stream] 🎯 T1 TRAIL {ticker} {sig['direction']} "
                f"price={price:.2f} pnl=+{pnl_pct:.2f}% regime=trending — riding to T2"
            )

        else:
            # ── CLOSE: book profit at T1, signal done ────────────────────────
            sb.table("signals").update({
                "status":        "closed",
                "result":        "win",
                "hit_target":    "t1",
                "result_pct":    round(pnl_pct, 4),
                "result_pnl":    round(pnl_abs, 4),
                "closed_reason": "target_hit",
                "closed_at":     datetime.now(timezone.utc).isoformat(),
            }).eq("id", sig["id"]).execute()

            sb.table("signal_events").insert({
                "signal_id":  sig["id"],
                "event_type": "closed_win",
                "price":      price,
                "note":       f"✅ T1 hit @ ${price:.2f} — profit booked +{pnl_pct:.1f}%",
            }).execute()

            try:
                strat_label = strategy.replace("_", " ")
                _push._send_raw(
                    title=f"✅ Book Profit — {ticker}  +{pnl_pct:.1f}%",
                    body=f"T1 hit on {sig['direction']} {strat_label}. Profit locked in at ${price:.2f}.",
                    data={"type": "signal_closed", "result": "win", "ticker": ticker,
                          "signal_id": str(sig["id"])},
                )
            except Exception:
                pass

            # Evict from cache immediately — prevents duplicate close on next tick
            sigs = _rt_cache.get(ticker, [])
            _rt_cache[ticker] = [s for s in sigs if s["id"] != sig["id"]]

            # Clear advisor state — signal is closed, no more advice needed
            try:
                from engine import signal_advisor as _advisor
                _advisor.evict(sig["id"])
            except Exception:
                pass

            logger.info(
                f"[stream] ✅ T1 BOOKED {ticker} {sig['direction']} "
                f"price={price:.2f} pnl=+{pnl_pct:.2f}% strategy={strategy}"
            )

    except Exception as e:
        logger.debug(f"[stream] RT T1 handler failed for {sig.get('id')}: {e}")


# ── Price momentum buffer helpers ─────────────────────────────────────────────

def _update_price_buffer(ticker: str, price: float) -> None:
    """
    Append the latest trade price to the rolling 60-second buffer for this ticker.
    Old entries (beyond _PRICE_BUFFER_WINDOW_S) are pruned on each write so the
    buffer stays memory-bounded even for high-frequency tickers.
    """
    now     = time.monotonic()
    buf     = _price_buffer.setdefault(ticker, [])
    buf.append((price, now))
    cutoff  = now - _PRICE_BUFFER_WINDOW_S
    # Prune in-place — keep only entries inside the look-back window
    _price_buffer[ticker] = [(p, t) for (p, t) in buf if t >= cutoff]


def _has_tick_momentum(ticker: str) -> bool:
    """
    Return True if price moved ≥ _MOMENTUM_THRESHOLD (0.4%) in the last 60 s.

    The buffer stores 10 minutes of history, but the trigger window for
    day_trade scans is always just the last 60 seconds so we don't fire
    on slow drifts that accumulated over many minutes.
    """
    buf = _price_buffer.get(ticker)
    if not buf or len(buf) < 2:
        return False
    now    = time.monotonic()
    cutoff = now - 60.0
    recent = [p for (p, t) in buf if t >= cutoff]
    if len(recent) < 2:
        return False
    lo, hi = min(recent), max(recent)
    if lo <= 0:
        return False
    return (hi - lo) / lo >= _MOMENTUM_THRESHOLD


# ── Tick-triggered day_trade processor ────────────────────────────────────────

def _process_daytrade_ticker_sync(symbol: str, price: float) -> None:
    """
    Run the full day_trade SMC pipeline for a single ticker, triggered by a
    live momentum event rather than a 15-min bar boundary.

    This is the mechanism that makes new day_trade signals fire within seconds
    of a setup forming instead of waiting up to 15 minutes for the next bar close.

    Called from _scan_executor — identical footprint to _process_bar_sync.
    Throttled to _TICK_DAYTRADE_THROTTLE_S (15 min) per ticker in on_trade.
    """
    try:
        session = _get_session()
        if session.get("blocked") or not session.get("market_open"):
            logger.debug(
                f"[stream] {symbol} tick day_trade skipped — "
                f"{session.get('block_reason', 'market closed')}"
            )
            return

        regime = _get_regime()
        if regime.get("blocked"):
            logger.debug(
                f"[stream] {symbol} tick day_trade skipped — "
                f"{regime.get('block_reason', 'regime blocked')}"
            )
            return

        logger.info(
            f"[stream] ⚡ Tick day_trade: {symbol} @ {price:.2f} "
            f"(momentum ≥{_MOMENTUM_THRESHOLD*100:.1f}% in 60s) | "
            f"{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )

        from engine.runner import _process_smc_ticker, _supabase
        sb     = _supabase()
        config = {"type": "day_trade", "interval": "15m", "period": "5d"}
        _process_smc_ticker(sb, symbol, config, regime=regime, session=session)

    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(
            f"[stream] Tick day_trade error for {symbol}: {e}", exc_info=True
        )


# ── Near-stop real-time warning ───────────────────────────────────────────────
# Fires a push notification + signal_events entry when price enters the danger
# zone (within 25% of stop distance from the SL level) on a live tick.
# This is the SAME threshold deriveStatus() uses on the app to show 🚨 "Near Stop".
# Throttled: at most once per 10 minutes per signal to avoid notification spam.
#
# Timeline:
#   Old behaviour: signal_monitor runs every 5-15 min → "Near Stop" up to 14 min late
#   New behaviour: _check_rt_levels runs every 1 second → warning fires within 1 second
_near_stop_warned: dict[str, float] = {}   # signal_id → last warned (monotonic)
_NEAR_STOP_THROTTLE_S = 600.0              # warn at most once per 10 minutes per signal


def _warn_near_stop(sig: dict, price: float) -> None:
    """
    Send a near-stop push notification and write a signal_events entry.
    Called at most once per _NEAR_STOP_THROTTLE_S seconds per signal.
    """
    try:
        sig_id  = sig["id"]
        ticker  = sig["ticker"]
        entry   = float(sig["entry_price"])
        sl      = float(sig["stop_loss"])
        is_long = sig["direction"] == "LONG"

        pnl_pct    = ((price - entry) / entry * 100) if is_long else ((entry - price) / entry * 100)
        dist_to_sl = abs(price - sl)
        dist_pct   = (dist_to_sl / price * 100) if price else 0   # true price distance

        # Clear wording: dollar + true price distance to the stop (not "% of the
        # stop buffer", which read as a price move and confused). No "act fast".
        note = (
            f"⚠️ Near Stop — ${price:.2f} · stop ${sl:.2f} "
            f"(${dist_to_sl:.2f} / {dist_pct:.1f}% away) · "
            f"P&L {'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%"
        )

        key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
        from supabase import create_client as _sc
        from engine import push as _push

        sb = _sc(os.environ["SUPABASE_URL"], key)
        sb.table("signal_events").insert({
            "signal_id":  sig_id,
            "event_type": "near_stop",
            "price":      price,
            "note":       note,
        }).execute()

        try:
            direction = sig["direction"]
            _push._send_raw(
                title=f"⚠️ Near Stop — {ticker}  {pnl_pct:+.1f}%",
                body=f"{direction} position: ${price:.2f} approaching stop ${sl:.2f}. Review now.",
                data={"type": "near_stop", "ticker": ticker, "signal_id": str(sig_id)},
            )
        except Exception:
            pass

        logger.info(
            f"[stream] ⚠️ NEAR-STOP {ticker} {sig['direction']} "
            f"price={price:.2f} sl={sl:.2f} pnl={pnl_pct:+.1f}% "
            f"({pct_of_stop:.0f}% from stop)"
        )
    except Exception as e:
        logger.debug(f"[stream] Near-stop warn failed for {sig.get('id')}: {e}")


def _is_rth_now() -> bool:
    """Regular US trading hours (9:30 AM–4:00 PM ET, Mon–Fri). Cheap + real-time
    (no calendar/network) — gates real-time EXITS so we don't close trades on
    thin, unfillable pre/post-market prints that routinely reverse at the open.
    Genuine gaps are still caught at the RTH open by the first in-session tick +
    the 5-min monitor backstop."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    n = datetime.now(ZoneInfo("America/New_York"))
    if n.weekday() >= 5:
        return False
    mins = n.hour * 60 + n.minute
    return 570 <= mins < 960   # 9:30 (570) .. 16:00 (960)


def _check_rt_levels(ticker: str, price: float) -> None:
    """
    Check a live trade price against all cached active signals for this ticker.
    Called at most once per second per ticker (throttled in on_trade).

    Decision tree per signal:
      T2 crossed       → close as win  (full target)
      T1 crossed       → move SL to breakeven, ride to T2 (don't close)
      SL crossed       → close as loss
      Near SL (≤25%)   → push warning (does NOT close) — throttled 1/10 min
      SL already at breakeven (T1 already hit) → only T2 / SL(=entry) matter
    """
    global _rt_cache, _rt_cache_ts

    # Regular-hours only: don't close target/stop on extended-hours prints (thin,
    # unfillable, and they whip back at the open). The 5-min monitor backstop is
    # already RTH-gated; a real gap is enforced at the open by the first RTH tick.
    if not _is_rth_now():
        return

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
                else:
                    # Price hasn't crossed any level — check near-stop danger zone.
                    # Same 25% threshold as deriveStatus() in the app so the push
                    # fires the exact moment the card pill turns 🚨 Near Stop.
                    stop_dist = abs(entry - sl)
                    if stop_dist > 0 and (price - sl) <= stop_dist * 0.25:
                        last_warn = _near_stop_warned.get(sig["id"], 0.0)
                        if time.monotonic() - last_warn >= _NEAR_STOP_THROTTLE_S:
                            _near_stop_warned[sig["id"]] = time.monotonic()
                            _rt_executor.submit(_warn_near_stop, sig.copy(), price)
            else:   # SHORT
                if price <= t2:
                    _close_rt_signal(sig, "t2", price)
                elif price <= t1 and not t1_already_hit:
                    _handle_t1_rt(sig, price)
                elif price >= sl:
                    _close_rt_signal(sig, "sl", price)
                else:
                    # SHORT near-stop: price is within 25% of stop distance above SL
                    stop_dist = abs(entry - sl)
                    if stop_dist > 0 and (sl - price) <= stop_dist * 0.25:
                        last_warn = _near_stop_warned.get(sig["id"], 0.0)
                        if time.monotonic() - last_warn >= _NEAR_STOP_THROTTLE_S:
                            _near_stop_warned[sig["id"]] = time.monotonic()
                            _rt_executor.submit(_warn_near_stop, sig.copy(), price)

            # ── Contextual hold/exit advisory ────────────────────────────────
            # Guard: only run if the signal wasn't evicted by a close above.
            # _close_rt_signal() removes the signal from _rt_cache immediately;
            # checking the cache here prevents advisor from running on a signal
            # that was just closed at T2 or SL this same tick.
            # (T1 trail path keeps the signal in cache — advisor still runs there,
            # which is desirable: momentum-toward-T2 advice is most relevant then.)
            sig_still_active = any(
                s["id"] == sig["id"] for s in _rt_cache.get(ticker, [])
            )
            if sig_still_active:
                try:
                    from engine import signal_advisor as _advisor
                    # Snapshot the full 10-min price history for this ticker.
                    # list() copy is GIL-safe — we don't hold a reference into
                    # the live buffer, so background pruning won't corrupt it.
                    price_hist = list(_price_buffer.get(ticker, []))
                    _advisor.check(sig, price, _get_regime(), _get_session(), price_hist)
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"[stream] RT level check error for {ticker}/{sig.get('id')}: {e}")


# ── Per-bar signal health checker ────────────────────────────────────────────
# Runs RSI/MACD (every 5-min bar close) and SMC structure reversal (every
# 15-min bar close) for any ticker that has active non-scalp signals.
# This moves the signal_monitor's analysis from a 15-min schedule to being
# event-driven — latency drops from up to 15 min to 1–5 min.
#
# Throttle: _bar_health_last prevents the same check running twice if multiple
# tickers happen to deliver bars at the same minute.
_bar_health_rsi_last:       dict[str, float] = {}   # ticker → monotonic
_bar_health_structure_last: dict[str, float] = {}   # ticker → monotonic
_BAR_HEALTH_RSI_THROTTLE_S       = 270.0   # 4.5 min — fires on 5-min bars
_BAR_HEALTH_STRUCTURE_THROTTLE_S = 870.0   # 14.5 min — fires on 15-min bars


def _check_signal_health_rsi(symbol: str, close: float) -> None:
    """
    RSI + MACD momentum check for all active signals on this ticker.
    Called every 5-min bar close. Writes a signal_events entry + push if
    momentum is failing. Throttled so it never duplicates within 4.5 min.
    """
    sigs = _rt_cache.get(symbol)
    if not sigs:
        return

    try:
        from engine.signal_monitor import _momentum_check, _log_event, _push_early_book
        import os
        from supabase import create_client as _sc

        key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
        sb  = _sc(os.environ["SUPABASE_URL"], key)

        for sig in list(sigs):
            try:
                direction = sig["direction"]
                entry     = float(sig["entry_price"])
                is_long   = direction == "LONG"
                pnl_pct   = (
                    (close - entry) / entry * 100
                    if is_long
                    else (entry - close) / entry * 100
                )

                # Only check when signal is meaningfully in profit — RSI/MACD
                # advice is only relevant when you have something to protect.
                if pnl_pct < 0.3:
                    continue

                book_now, reason = _momentum_check(symbol, direction)
                if book_now:
                    note = (
                        f"📊 {reason} — P&L: {pnl_pct:+.1f}% @ ${close:.2f}. "
                        f"Consider booking profit."
                    )
                    _log_event(sb, sig["id"], "advisor_momentum", price=close, note=note)
                    try:
                        from engine import push as _push
                        _push._send_raw(
                            title=f"📊 Momentum Signal — {symbol}  {pnl_pct:+.1f}%",
                            body=reason,
                            data={"type": "advisor_momentum", "ticker": symbol,
                                  "signal_id": str(sig["id"])},
                        )
                    except Exception:
                        pass
                    logger.info(
                        f"[stream] 📊 RSI/MACD health: {symbol} {direction} "
                        f"pnl={pnl_pct:+.1f}% — {reason}"
                    )

            except Exception as _se:
                logger.debug(f"[stream] RSI health inner error {symbol}: {_se}")

    except Exception as e:
        logger.debug(f"[stream] RSI health check error for {symbol}: {e}")


def _check_signal_health_structure(symbol: str, close: float) -> None:
    """
    SMC structure reversal (CHoCH) check for all active signals on this ticker.
    Called every 15-min bar close. Fires a push + event if the structure has
    flipped against the signal direction. Does NOT auto-close — advice only.
    """
    sigs = _rt_cache.get(symbol)
    if not sigs:
        return

    try:
        from engine.signal_monitor import _detect_structure_reversal, _log_event
        import os
        from supabase import create_client as _sc

        key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
        sb  = _sc(os.environ["SUPABASE_URL"], key)

        for sig in list(sigs):
            try:
                direction = sig["direction"]
                entry     = float(sig["entry_price"])
                is_long   = direction == "LONG"
                pnl_pct   = (
                    (close - entry) / entry * 100
                    if is_long
                    else (entry - close) / entry * 100
                )

                if _detect_structure_reversal(symbol, direction):
                    opposite = "bearish" if direction == "LONG" else "bullish"
                    note = (
                        f"⚠️ {opposite.capitalize()} CHoCH on 15m chart — "
                        f"structure reversed against {direction}. "
                        f"P&L: {pnl_pct:+.1f}% @ ${close:.2f}. Consider exiting."
                    )
                    _log_event(sb, sig["id"], "advisor_reversal",
                               price=close, note=note)
                    try:
                        from engine import push as _push
                        _push._send_raw(
                            title=f"🔄 Structure Reversed — {symbol}  {pnl_pct:+.1f}%",
                            body=(
                                f"{opposite.capitalize()} CHoCH detected on 15m. "
                                f"{direction} thesis may be invalidated — review position."
                            ),
                            data={"type": "advisor_reversal", "ticker": symbol,
                                  "signal_id": str(sig["id"])},
                        )
                    except Exception:
                        pass
                    logger.info(
                        f"[stream] 🔄 Structure reversal: {symbol} {direction} "
                        f"— {opposite} CHoCH @ ${close:.2f}"
                    )

            except Exception as _se:
                logger.debug(f"[stream] Structure health inner error {symbol}: {_se}")

    except Exception as e:
        logger.debug(f"[stream] Structure health check error for {symbol}: {e}")


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

    global _subscribed_tickers, _pending_tickers, _wss_ref, _on_trade_ref, _on_bar_ref

    from engine.runner import ALL_TICKERS, SCALP_TICKERS
    scalp_set = set(SCALP_TICKERS)

    # Subscribe to all watched tickers — 1-min bars serve as clock ticks
    # for 5m (scalp), 15m (day_trade/flow), and 1h (swing) boundary detection.
    all_subscribe = list(dict.fromkeys(ALL_TICKERS))   # preserve order, deduplicate

    # Also subscribe tickers from currently-active signals so their stops are
    # checked tick-by-tick (otherwise we'd rely on the 5-min signal_monitor and
    # could miss tight stops by hundreds of bps — see AAL incident 2026-05-27).
    try:
        from supabase import create_client
        _sb = create_client(os.environ["SUPABASE_URL"], os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_KEY", ""))
        _rows = _sb.table("signals").select("ticker").eq("status", "active").execute().data or []
        active_tickers = sorted({r["ticker"] for r in _rows if r.get("ticker")})
        if active_tickers:
            extra = [t for t in active_tickers if t not in all_subscribe]
            all_subscribe.extend(extra)
            logger.info(f"[stream] Subscribing {len(extra)} extra ticker(s) from active signals: {extra}")
    except Exception as _e:
        logger.warning(f"[stream] Could not load active-signal tickers on startup: {_e}")

    # Restore per-tick detector zones staged before this restart so a deploy
    # doesn't reset compression/pullback/swing to zero — AND re-subscribe their
    # tickers so the restored zones are actually monitored (receive 1m bars),
    # not just sitting in memory until the next scan re-arms them.
    try:
        zone_tickers = load_zones_from_db()
        extra_z = [t for t in zone_tickers if t not in all_subscribe]
        if extra_z:
            all_subscribe.extend(extra_z)
            logger.info(f"[stream] Re-subscribing {len(extra_z)} ticker(s) from restored zones")
    except Exception as _ze:
        logger.warning(f"[stream] zone restore/resubscribe failed: {_ze}")

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

                # NOTE: armed zones are NOT cleared at the close — they're kept
                # through after-hours for analysis. A scheduled job clears them
                # ~00:30 ET (runner._clear_zones_overnight) so the next session
                # starts fresh.

                # ── EVERY bar: check scalp T1/SL in real-time ─────────────────
                # Uses bar high/low (wicks) so we catch levels touched intra-bar.
                # Runs before any boundary logic so closes fire as fast as possible.
                # Uses _rt_executor (dedicated) so strategy scans never delay this.
                _rt_executor.submit(_check_scalp_levels, symbol, bar_high, bar_low)

                # ── Per-1m-CLOSE breakout/reclaim confirmation ────────────────
                # Compression / pullback / swing-breakout fire here (not per-tick)
                # so they require the 1-minute bar to CLOSE beyond the level —
                # body confirmation. A tall wick that retraces by the close no
                # longer triggers a false breakout. `close` is the bar body, not
                # an intrabar high/low.
                try:
                    _check_compression_breakout(symbol, close, bar_high, bar_low, volume)
                except Exception:
                    pass
                try:
                    _check_pullback_reclaim(symbol, close, bar_high, bar_low, volume)
                except Exception:
                    pass
                try:
                    _check_swing_breakout(symbol, close, bar_high, bar_low, volume)
                except Exception:
                    pass
                # Retest fills for breakouts armed on a prior bar
                try:
                    _check_retest(symbol, close, bar_high, bar_low)
                except Exception:
                    pass
                # Record this bar's volume AFTER the checks (so the surge test
                # compares the confirming bar against PRIOR bars, not itself).
                _record_volume(symbol, volume)

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

                # ── Per-bar signal health: RSI/MACD every 5-min bar ───────────
                # Only for tickers that currently have active signals in the RT
                # cache — no point running for tickers nobody is holding.
                # Throttle prevents duplicate runs if bar events arrive in burst.
                if symbol in _rt_cache and minute % 5 == 0:
                    now_mono = time.monotonic()
                    if (now_mono - _bar_health_rsi_last.get(symbol, 0.0)
                            >= _BAR_HEALTH_RSI_THROTTLE_S):
                        _bar_health_rsi_last[symbol] = now_mono
                        _scan_executor.submit(_check_signal_health_rsi, symbol, close)

                # ── Per-bar signal health: structure reversal every 15-min bar ─
                # CHoCH detection runs SMC on 15m bars — heavier than RSI but
                # gives the clearest reversal signal. Rate-matched to bar frequency.
                if symbol in _rt_cache and minute % 15 == 0:
                    now_mono = time.monotonic()
                    if (now_mono - _bar_health_structure_last.get(symbol, 0.0)
                            >= _BAR_HEALTH_STRUCTURE_THROTTLE_S):
                        _bar_health_structure_last[symbol] = now_mono
                        _scan_executor.submit(
                            _check_signal_health_structure, symbol, close
                        )

            # ── Trade handler: price broadcast + real-time level checks ──────
            async def on_trade(trade) -> None:
                ticker = trade.symbol
                price  = float(trade.price)
                size   = float(getattr(trade, "size", 0) or 0)

                # 1. Feed price to WebSocket clients (always — no throttle)
                try:
                    from engine.price_store import update as price_update
                    price_update(ticker, price)
                except Exception:
                    pass   # never let price broadcast errors kill the stream

                # 1b. Feed trade tape (block prints, tape acceleration, VWAP).
                #     Pure in-memory, very fast. See engine/trade_tape.py.
                try:
                    from engine import trade_tape
                    trade_tape.record_trade(ticker, price, size)
                except Exception:
                    pass

                # NOTE: breakout/reclaim confirmation moved OFF the per-tick path
                # (2026-05-28). Firing on a single trade tick crossing the level
                # fired on intrabar WICKS that immediately retraced (NVDA false
                # positive). The three _check_* detectors now run in on_bar on the
                # 1-minute CLOSE (body confirmation) so a wick alone can't trigger.

                # 2. Real-time T1/T2/SL check for ALL active non-scalp signals.
                #    Throttled to at most once per second per ticker so we don't
                #    flood the executor with thousands of tasks on liquid stocks.
                now = time.monotonic()
                if now - _rt_last_check.get(ticker, 0.0) >= _RT_THROTTLE_S:
                    _rt_last_check[ticker] = now
                    # _rt_executor: dedicated pool — never queued behind a scan
                    _rt_executor.submit(_check_rt_levels, ticker, price)

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

                # 4. Price momentum buffer — rolling 60-second window per ticker.
                #    Updated on every tick so _has_tick_momentum() has fresh data.
                #    Runs for ALL tickers (scalp and non-scalp alike) so the buffer
                #    is always populated regardless of strategy.
                _update_price_buffer(ticker, price)

                # 5. Tick-triggered day_trade scan — fires when price moves ≥0.4%
                #    in the last 60 seconds, bypassing the 15-min bar-close boundary.
                #    New day_trade signals can now appear within seconds of a setup
                #    forming rather than waiting up to 15 minutes.
                #    Throttle: 900 s (15 min) per ticker — same cadence as bar-close
                #    scans so we never flood the pipeline with redundant work.
                if _has_tick_momentum(ticker):
                    if now - _tick_daytrade_last.get(ticker, 0.0) >= _TICK_DAYTRADE_THROTTLE_S:
                        _tick_daytrade_last[ticker] = now
                        _scan_executor.submit(_process_daytrade_ticker_sync, ticker, price)

            wss.subscribe_bars(on_bar, *all_subscribe)
            wss.subscribe_trades(on_trade, *all_subscribe)

            # ── Register live references for dynamic ticker subscriptions ──
            _wss_ref      = wss
            _on_trade_ref = on_trade
            _on_bar_ref   = on_bar

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
            _on_bar_ref   = None
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
            _on_bar_ref   = None
            logger.info("[stream] Stream task cancelled — shutting down")
            _scan_executor.shutdown(wait=False)
            _rt_executor.shutdown(wait=False)
            return

        except Exception as e:
            _wss_ref      = None
            _on_trade_ref = None
            _on_bar_ref   = None
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
