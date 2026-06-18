"""
Trend-ride management for confirmed-green SWING signals.

WHY (the HOOD post-mortem, Jun 2026): the engine kept exiting working swing
trends early — structure_reversal on a routine 15m pullback, market_close at the
bell, and a tight 2.5%-below-peak trail. HOOD ran $73→$110 (+50%) and we caught it
five times yet captured ~scratch every time, because we managed a multi-week swing
like a day-trade. This module lets a swing that is genuinely TRENDING ride:

  ACTIVATE (per analyst gate) when ALL hold:
    • the signal is a swing (daily timeframe or a swing strategy), AND
    • it is GREEN (price beyond entry in the trade's favor), AND
    • the last COMPLETED daily close is on the right side of a RISING 20-day MA
      (LONG: close ≥ MA and MA rising; SHORT: close ≤ MA and MA falling).

  WHILE RIDING:
    • the caller SUPPRESSES the early exits (structure_reversal, intelligent-exit,
      tight peak-trail, near-expiry book) and does NOT market-close it,
    • the hard stop is trailed up UNDER the recent daily SWING LOW (not at the MA —
      a stop at the MA gets wicked out intraday; HOOD 06-10 wicked to $84 but CLOSED
      $86), ratcheting in the trade's favour only.

  EXIT the ride on a DECISIVE signal — the last COMPLETED daily close crossing back
  through the 20-MA (LONG: close < MA; SHORT: close > MA). Intraday wicks never exit.

The hard catastrophic stop + T1/T2 are still enforced by signal_monitor's backstop —
this only governs the EARLY-exit behaviour. Gated behind TREND_RIDE_ENABLED (kill
switch) and tagged on the signal (score_breakdown.trend_ride) so its effect is
measurable and reversible.
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from engine import smc

_ET = ZoneInfo("America/New_York")

# ── Kill switch ────────────────────────────────────────────────────────────────
def enabled() -> bool:
    return os.environ.get("TREND_RIDE_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

# ── What counts as a swing (mirror signal_monitor._SWING_LIKE_STRATEGIES) ───────
_SWING_STRATEGIES = {
    "swing_trade", "breakdown", "breakout", "turnaround", "peak",
    "breakdown_forming", "distrib_forming", "peak_forming", "turn_forming",
    "accum_forming", "position_trade",
}
_DAILY_TFS = {"1day", "1d", "d", "daily"}

_MA_PERIOD       = 20
_SLOPE_LOOKBACK  = 5      # MA must be higher than it was this many completed bars ago
_SWING_LOOKBACK  = 3      # trail the stop under the lowest low of the last N completed days
_BUF_ATR_FRAC    = 0.25   # stop buffer below the swing low, in ATRs
_CTX_TTL_S       = 240.0  # daily bars only change once/day → cache within a monitor pass

_ctx_cache: dict[str, tuple[float, dict | None]] = {}


def is_swing(sig: dict) -> bool:
    tf = str(sig.get("timeframe") or "").strip().lower()
    strat = (sig.get("strategy_type") or "").strip()
    return tf in _DAILY_TFS or strat in _SWING_STRATEGIES


def reset_cache() -> None:
    """Call at the start of each monitor pass (optional; the TTL also bounds staleness)."""
    _ctx_cache.clear()


def _completed_daily(df):
    """Drop today's still-forming bar so the 20-MA, slope, swing low and the
    close-vs-MA exit all use only COMPLETED sessions (no intraday wick leakage)."""
    if df is None or len(df) == 0:
        return df
    try:
        last_date = df.index[-1].date()
        if last_date == datetime.now(_ET).date():
            return df.iloc[:-1]
    except Exception:
        pass
    return df


def daily_context(ticker: str) -> dict | None:
    """Completed-bar daily context for the trend-ride decision, cached per ticker.
    Returns {ma20, ma20_prev, last_close, swing_low, swing_high, atr} or None."""
    now = time.time()
    hit = _ctx_cache.get(ticker)
    if hit and (now - hit[0]) < _CTX_TTL_S:
        return hit[1]
    ctx: dict | None = None
    try:
        df = smc.fetch_candles(ticker, period="3mo", interval="1d")
        df = _completed_daily(df)
        if df is not None and len(df) >= _MA_PERIOD + _SLOPE_LOOKBACK:
            close = df["close"]
            ma = close.rolling(_MA_PERIOD).mean()
            ctx = {
                "ma20":       float(ma.iloc[-1]),
                "ma20_prev":  float(ma.iloc[-1 - _SLOPE_LOOKBACK]),
                "last_close": float(close.iloc[-1]),
                "swing_low":  float(df["low"].tail(_SWING_LOOKBACK).min()),
                "swing_high": float(df["high"].tail(_SWING_LOOKBACK).max()),
                "atr":        float((df["high"] - df["low"]).tail(14).mean()),
            }
    except Exception:
        ctx = None
    _ctx_cache[ticker] = (now, ctx)
    return ctx


def evaluate(sig: dict, price: float, ctx: dict) -> dict:
    """Pure decision. Returns:
      active     — ride/keep-riding (suppress early exits, trail under swing low)
      break_exit — was riding and the daily close decisively crossed back through the MA → exit
      trail_sl   — proposed hard stop (under the recent swing low ± buffer); caller ratchets
      ma20, last_close — for logging
    """
    is_long   = sig.get("direction") == "LONG"
    entry     = float(sig.get("entry_price") or 0)
    ma20      = ctx["ma20"]; ma20_prev = ctx["ma20_prev"]; last_close = ctx["last_close"]
    atr       = ctx.get("atr") or 0.0
    buf       = max(atr * _BUF_ATR_FRAC, (price or last_close) * 0.002)
    was_riding = bool(((sig.get("score_breakdown") or {}).get("trend_ride")))

    green       = (price > entry) if is_long else (price < entry and entry > 0)
    ma_rising   = (ma20 >= ma20_prev) if is_long else (ma20 <= ma20_prev)
    above_trend = (last_close >= ma20) if is_long else (last_close <= ma20)

    active     = bool(green and ma_rising and above_trend)
    # Exit only a trade that WAS riding — a fresh green signal that hasn't cleared the
    # MA yet just falls through to normal management (never trend-closed prematurely).
    break_exit = bool(was_riding and green and (last_close < ma20 if is_long else last_close > ma20))

    if is_long:
        trail_sl = round(ctx["swing_low"] - buf, 2)
    else:
        trail_sl = round(ctx["swing_high"] + buf, 2)

    return {
        "active":     active,
        "break_exit": break_exit,
        "was_riding": was_riding,
        "trail_sl":   trail_sl,
        "ma20":       round(ma20, 2),
        "last_close": round(last_close, 2),
    }
