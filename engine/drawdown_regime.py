"""
Market drawdown-regime detector — Phase 0 of the crash/deep-value signal (#10).
================================================================================
The "WHEN to buy" half: detects when the broad market is deeply drawn down off
its 52-week high — historically the accumulation window for quality names (paired
with engine/fundamentals.py, the "WHICH names" half).

compute() is PURE (given index daily bars) → unit-testable. assess() does the IO
(SPY/QQQ/IWM via Alpaca) and caches for 30 min.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger("signalbolt.drawdown_regime")

# Grade thresholds — SPY (broad market) % off its 52-week high.
#   healthy > -5 · pullback -5..-10 · correction -10..-20 · bear -20..-30 · deep <= -30
# accumulation_window (the buy trigger) opens at bear (<= -20%); deep at <= -30%.
_ACCUMULATION_AT = -20.0
_WATCH_AT        = -10.0
_DEEP_AT         = -30.0

_INDICES = ("SPY", "QQQ", "IWM")


def _off_high(df) -> dict | None:
    """{off_high_pct, high_52w, last} for one index, or None if insufficient bars."""
    if df is None or len(df) < 30 or "high" not in df.columns or "close" not in df.columns:
        return None
    window = df.tail(252)                      # ~52 weeks of trading days
    hi = float(window["high"].max())
    last = float(df["close"].iloc[-1])
    if hi <= 0:
        return None
    return {"off_high_pct": round((last / hi - 1) * 100, 1),
            "high_52w": round(hi, 2), "last": round(last, 2)}


def compute(index_bars: dict) -> dict:
    """index_bars: {symbol: daily-bars df}. Returns the regime classification."""
    per = {}
    for sym, df in (index_bars or {}).items():
        d = _off_high(df)
        if d:
            per[sym] = d
    if not per:
        return {"regime": "unknown", "label": "No data", "off_high_pct": None,
                "accumulation_window": False, "watch": False, "deep": False, "indices": {}}

    # SPY is the broad-market reference; if missing, use the worst available.
    primary = per.get("SPY") or min(per.values(), key=lambda x: x["off_high_pct"])
    off = primary["off_high_pct"]

    if off <= _DEEP_AT:
        regime, label = "deep_bear", "Deep bear — generational accumulation window"
    elif off <= _ACCUMULATION_AT:
        regime, label = "bear", "Bear market — strong accumulation window"
    elif off <= _WATCH_AT:
        regime, label = "correction", "Correction — watch for accumulation"
    elif off <= -5.0:
        regime, label = "pullback", "Pullback — normal"
    else:
        regime, label = "healthy", "Healthy — near highs"

    return {
        "regime": regime,
        "label": label,
        "off_high_pct": off,
        "accumulation_window": off <= _ACCUMULATION_AT,
        "watch": off <= _WATCH_AT,
        "deep": off <= _DEEP_AT,
        "indices": per,
    }


# ── IO + 30-min cache ──────────────────────────────────────────────────────
_cache: dict | None = None
_cache_ts = 0.0


def assess(force: bool = False) -> dict:
    """Fetch SPY/QQQ/IWM and classify. Cached 30 min (regime is a daily-scale state)."""
    global _cache, _cache_ts
    if not force and _cache is not None and (time.time() - _cache_ts) < 1800:
        return _cache
    try:
        from engine import alpaca_client as ac
        bars = {sym: ac.get_bars(sym, "1Day", days=400) for sym in _INDICES}
        res = compute(bars)
    except Exception as e:
        logger.warning(f"[drawdown_regime] assess failed: {e}")
        res = {"regime": "unknown", "label": "No data", "off_high_pct": None,
               "accumulation_window": False, "watch": False, "deep": False, "indices": {}}
    _cache, _cache_ts = res, time.time()
    return res
