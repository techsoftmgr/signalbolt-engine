"""
entry_guards.py — ENTRY-side guards from the SNOW / GLD analysis (2026-08).

1. RE-FIRE / DEDUP SUPPRESSION (`should_suppress`): don't stack a same ticker+direction
   idea. Blocks (a) a new signal when one is already ACTIVE (GLD fired 13 near-identical
   longs in 2 days), and (b) re-firing within a cooldown after a STOP-OUT (SNOW was
   shorted ~11x as it ran +21%).

2. SHORT-INTO-STRENGTH VETO (`short_into_strength`): don't SHORT a strong-uptrend leader —
   price above a RISING 20-EMA, within 3% of the 20-day high, outperforming SPY. Wait for
   a confirmed rollover instead of shorting strength (SNOW peak-shorts into new highs).

Both are kill-switched (default OFF), fail-open (never worse than today's engine), and
their blocks are logged to entry_gate_rejections so the gate-validator can measure whether
they killed losers (good) or winners (bad) before we trust them.
"""
from __future__ import annotations
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger("signalbolt.guards")


def _flag(name: str) -> bool:
    return os.environ.get(name, "false").strip().lower() in ("1", "true", "yes", "on")


def cooldown_enabled() -> bool:   return _flag("REFIRE_COOLDOWN_ENABLED")
def short_veto_enabled() -> bool: return _flag("SHORT_STRENGTH_VETO_ENABLED")

_COOLDOWN_DAYS = int(os.environ.get("REFIRE_COOLDOWN_DAYS", "3"))


def should_suppress(sb, ticker: str, direction: str, days: int = 0) -> tuple[bool, str]:
    """True → don't fire. Covers (a) an already-ACTIVE same ticker+direction signal (dedup)
    and (b) a same ticker+direction STOP-OUT within the cooldown window. Fail-open."""
    d = days or _COOLDOWN_DAYS
    try:
        active = (sb.table("signals").select("id")
                  .eq("ticker", ticker).eq("direction", direction).eq("status", "active")
                  .limit(1).execute().data) or []
        if active:
            return True, f"{ticker} {direction} already active — dedup"
        since = (datetime.now(timezone.utc) - timedelta(days=d)).isoformat()
        closed = (sb.table("signals").select("closed_reason,result_pct")
                  .eq("ticker", ticker).eq("direction", direction).eq("status", "closed")
                  .gte("closed_at", since).order("closed_at", desc=True).limit(10)
                  .execute().data) or []
        for r in closed:
            if r.get("closed_reason") == "stop_hit" or (r.get("result_pct") or 0) < -0.1:
                return True, f"{ticker} {direction} stopped out within {d}d — cooldown"
        return False, ""
    except Exception as e:
        logger.debug(f"[guards] suppress check failed {ticker}/{direction}: {e}")
        return False, ""


def short_into_strength(ticker: str, daily_df: Optional[pd.DataFrame],
                        spy_df: Optional[pd.DataFrame] = None) -> tuple[bool, str]:
    """Veto a SHORT on a strong-uptrend leader: price above a RISING 20-EMA AND within 3%
    of the 20-day high AND outperforming SPY over 5d. Pure; fail-open (False) on thin data.
    A predictive peak-short should wait for the uptrend STRUCTURE to break, not fire into
    new highs (SNOW: shorted 11x while +21%)."""
    try:
        if daily_df is None or len(daily_df) < 25:
            return False, ""
        close = daily_df["close"].astype(float)
        c = float(close.iloc[-1])
        ema20 = close.ewm(span=20, adjust=False).mean()
        e_now, e_prev = float(ema20.iloc[-1]), float(ema20.iloc[-6])
        above_rising = (c > e_now) and (e_now > e_prev)
        hi20 = float(daily_df["high"].astype(float).iloc[-20:].max())
        near_high = c >= hi20 * 0.97
        rs_ok = True
        if spy_df is not None and len(spy_df) >= 6:
            sret = float(spy_df["close"].iloc[-1]) / float(spy_df["close"].iloc[-6]) - 1
            tret = c / float(close.iloc[-6]) - 1
            rs_ok = tret > sret
        if above_rising and near_high and rs_ok:
            return True, (f"{ticker} above rising 20-EMA ({c:.2f}>{e_now:.2f}), near 20d-high "
                          f"({hi20:.2f}), RS>SPY — not shorting strength")
        return False, ""
    except Exception:
        return False, ""


_spy_cache: dict = {"df": None, "ts": 0.0}


def _spy_daily() -> Optional[pd.DataFrame]:
    if _spy_cache["df"] is not None and (time.monotonic() - _spy_cache["ts"]) < 3600:
        return _spy_cache["df"]
    try:
        from engine.alpaca_client import get_bars
        df = get_bars("SPY", "1Day", 90)
        if df is not None and len(df):
            _spy_cache["df"], _spy_cache["ts"] = df, time.monotonic()
        return df
    except Exception:
        return _spy_cache["df"]


def short_into_strength_check(ticker: str) -> tuple[bool, str]:
    """Fetch daily + SPY and evaluate the short-into-strength veto (SPY cached ~1h)."""
    try:
        from engine.alpaca_client import get_bars
        return short_into_strength(ticker, get_bars(ticker, "1Day", 90), _spy_daily())
    except Exception:
        return False, ""
