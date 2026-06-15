"""
Market Pulse — thrust / breakdown PLAYBOOK.

When the rare breadth THRUST (bullish) or BREAKDOWN (bearish) fires, the daily
read shifts the *backdrop* — it is NOT a prompt to buy/short this candle or any
specific ticker. To make that usable instead of abstract, this module spells out:

  • plain-English "how to use it" guidance (wait for a setup, enter with a stop),
  • the ACTUAL reference levels on the broad-market proxies (SPY + QQQ): last
    close, the 9-day EMA, the 20-day EMA, and the concrete 9/20 reclaim (thrust)
    or loss (breakdown) price — so the user never has to compute it themselves,
  • an explicit not-a-recommendation line.

Levels are "as of the last settled close" (today's forming bar is dropped before
4pm ET, mirroring the daily job) so they line up with the EOD thrust/breakdown
read that surfaces them. SPY/QQQ are MARKET reference points, never a per-ticker
call — that framing is the whole point.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from . import data

logger = logging.getLogger("signalbolt.market_pulse.playbook")

_HOW_TO = {
    "thrust": [
        "Green light to be aggressive on longs — the broad-market backdrop just turned favorable, and historically that tailwind lasts weeks, not just today.",
        "This does NOT mean buy right now. A thrust fires AFTER a sharp run off the lows, so buying the moment it prints usually means chasing an extended move.",
        "Wait for a setup: a pullback that holds, a base/consolidation, or a 9/20 reclaim — a daily close back above both the 9- and 20-day EMA. Enter there, with a stop just below the level that held.",
    ],
    "breakdown": [
        "Defense first — the broad-market backdrop just turned hostile for longs. Common response: trim, raise cash, tighten stops, and demand A+ setups.",
        "This does NOT mean short right now. A breakdown often marks capitulation, so shorting the moment it prints can get steamrolled by the snapback.",
        "If you act bearish, wait for a setup: a lower high, a failed retest of the 9/20 zone, or a 9/20 loss — a daily close back below both EMAs. Enter there, with a tight stop above the level that rejected.",
    ],
}

_NOT_ADVICE = {
    "thrust": ("Not a prompt to buy this candle or any specific ticker — it's a market "
               "backdrop. The SPY/QQQ levels below are reference points, not a "
               "recommendation. Always combine with your own setup and risk management."),
    "breakdown": ("Not a prompt to short this candle or any specific ticker — it's a market "
                  "backdrop. The SPY/QQQ levels below are reference points, not a "
                  "recommendation. Always combine with your own setup and risk management."),
}

_TITLE = {
    "thrust": "Breadth thrust — bullish backdrop",
    "breakdown": "Breadth breakdown — defensive backdrop",
}


def classify(direction: str, last: float, ema9: float, ema20: float) -> dict:
    """Pure: where is price relative to the 9/20 zone, and what's the trigger?

    The "zone" is bounded by the two EMAs (low = nearer, high = farther). For a
    THRUST the actionable event is a *reclaim* (close above the higher EMA); for a
    BREAKDOWN it's a *loss* (close below the lower EMA). Returns raw (unrounded)
    floats; the caller rounds for display."""
    zlow, zhigh = (ema9, ema20) if ema9 <= ema20 else (ema20, ema9)
    if direction == "thrust":
        status = "above" if last >= zhigh else ("in_zone" if last >= zlow else "below")
        trigger, trigger_label = zhigh, "Reclaim level (daily close above)"
    else:  # breakdown
        status = "below" if last <= zlow else ("in_zone" if last <= zhigh else "above")
        trigger, trigger_label = zlow, "Loss level (daily close below)"
    return {"status": status, "zlow": zlow, "zhigh": zhigh,
            "trigger": trigger, "trigger_label": trigger_label}


def _note(direction: str, sym: str, last: float, ema9: float, ema20: float, c: dict) -> str:
    """One plain-English line per index, with the actual numbers embedded."""
    lo, hi = round(c["zlow"], 2), round(c["zhigh"], 2)
    e9, e20, px = round(ema9, 2), round(ema20, 2), round(last, 2)
    if direction == "thrust":
        if c["status"] == "above":
            return (f"{sym} {px} is already above its 9-EMA ({e9}) and 20-EMA ({e20}) — the reclaim "
                    f"has happened. Wait for a pullback into the {lo}–{hi} zone that holds, then enter "
                    f"with a stop below {lo}.")
        if c["status"] == "in_zone":
            return (f"{sym} {px} is between its 9-EMA and 20-EMA ({lo}–{hi}). A reclaim = a daily close "
                    f"back above {hi}. Enter on that; stop below {lo}.")
        return (f"{sym} {px} is below both EMAs. Reclaim trigger = a daily close back above {hi} "
                f"(the 9/20 zone is {lo}–{hi}). No long until then; stop below {lo}.")
    # breakdown
    if c["status"] == "below":
        return (f"{sym} {px} is already below its 9-EMA ({e9}) and 20-EMA ({e20}) — the 9/20 is lost. "
                f"Watch for a failed retest of the {lo}–{hi} zone (a lower high) to act on weakness; "
                f"tight stop above {hi}.")
    if c["status"] == "in_zone":
        return (f"{sym} {px} is between its 9-EMA and 20-EMA ({lo}–{hi}). A confirmed loss = a daily "
                f"close back below {lo}. Act on that; stop above {hi}.")
    return (f"{sym} {px} is still above both EMAs. Loss trigger = a daily close below {lo} "
            f"(the 9/20 zone is {lo}–{hi}). No bearish entry until then; stop above {hi}.")


def _drop_forming(df: pd.DataFrame) -> pd.DataFrame:
    """Drop today's still-forming daily bar before 4pm ET, so levels are 'as of the
    last settled close' (mirrors job.run_daily)."""
    from datetime import datetime as _dt, timezone as _tz
    try:
        from zoneinfo import ZoneInfo as _ZI
        now_et = _dt.now(_ZI("America/New_York"))
    except Exception:
        now_et = _dt.now(_tz.utc)
    last_date = pd.Timestamp(df.index[-1]).date()
    if last_date == now_et.date() and now_et.hour < 16:
        cutoff = pd.Timestamp(now_et.date()).tz_localize("UTC")
        return df[df.index < cutoff]
    return df


def _levels_for(sym: str, direction: str) -> Optional[dict]:
    try:
        df = data.index_bars(sym, days=80)
        if df is None or "close" not in df or len(df) < 21:
            return None
        df = _drop_forming(df)
        if len(df) < 21:
            return None
        close = df["close"].astype(float)
        ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        last = float(close.iloc[-1])
        c = classify(direction, last, ema9, ema20)
        return {
            "symbol": sym,
            "last": round(last, 2), "ema9": round(ema9, 2), "ema20": round(ema20, 2),
            "zone_low": round(c["zlow"], 2), "zone_high": round(c["zhigh"], 2),
            "trigger": round(c["trigger"], 2), "trigger_label": c["trigger_label"],
            "status": c["status"],
            "as_of": pd.Timestamp(df.index[-1]).date().isoformat(),
            "note": _note(direction, sym, last, ema9, ema20, c),
        }
    except Exception as e:
        logger.debug(f"[playbook] levels {sym} failed: {e}")
        return None


def build(direction: str) -> Optional[dict]:
    """Full playbook payload for a fired thrust/breakdown, or None if levels can't be
    computed (fails open — the banner then just shows its headline)."""
    if direction not in ("thrust", "breakdown"):
        return None
    levels = [lv for sym in ("SPY", "QQQ") if (lv := _levels_for(sym, direction))]
    if not levels:
        return None
    return {
        "direction": direction,
        "title": _TITLE[direction],
        "how_to": _HOW_TO[direction],
        "levels": levels,
        "as_of": levels[0].get("as_of"),
        "not_advice": _NOT_ADVICE[direction],
    }
