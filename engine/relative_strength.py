"""
Relative-strength leader check — the RS exemption to the regime long-veto.

The entry gate blanket-blocks new LONGs in bearish/risk-off regimes
("market regime RISK_OFF against LONG"). But a backtest (2y, 858 pullback
events) showed that LONGs on names with relative strength are +EV even in a
weak market (+0.24R, ~44% win @2R — on par with the same setup in a healthy
market at +0.22R), while NON-RS longs in a weak market are sharply -EV
(-0.79R). So the blanket veto throws away a profitable subset.

This module answers: "is this name strong enough to buy despite the weak
market?" — defined exactly as the study's `highRS` + uptrend-intact split:
  • outperforming SPY over the last 20 trading days, AND
  • daily 20-EMA rising, AND
  • price above its 50-SMA (longer-term uptrend intact)

Fails CLOSED (returns False) on any data gap, so a failure keeps the
protective veto in place rather than letting a long through unchecked.

Kill switch: RS_EXEMPTION_ENABLED (default on). PANIC is intentionally NOT
exempted by the caller — acute crashes flush even leaders.
"""
import logging
import os
import time

import numpy as np

logger = logging.getLogger("signalbolt.relative_strength")

_RET_LOOKBACK = 20      # trading days for the relative-return comparison
_EMA_SLOPE_BARS = 5     # 20-EMA must be higher than it was this many bars ago

# SPY daily bars barely change intraday — cache them so the exemption check
# doesn't refetch the benchmark on every blocked long.
_spy_cache = None
_spy_ts = 0.0
_SPY_TTL = 1800.0       # 30 min


def enabled() -> bool:
    """Kill switch — default ON. Set RS_EXEMPTION_ENABLED=false to disable."""
    return os.environ.get("RS_EXEMPTION_ENABLED", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _ret(closes: np.ndarray, lookback: int) -> float | None:
    if len(closes) < lookback + 1:
        return None
    base = float(closes[-(lookback + 1)])
    if base <= 0:
        return None
    return (float(closes[-1]) - base) / base


def _spy_ret20() -> float | None:
    """20-day SPY return, cached. None if unavailable."""
    global _spy_cache, _spy_ts
    if _spy_cache is not None and (time.monotonic() - _spy_ts) < _SPY_TTL:
        return _spy_cache
    try:
        from engine.alpaca_client import get_bars
        df = get_bars("SPY", timeframe="1Day", days=60)
        if df is None or len(df) < _RET_LOOKBACK + 1:
            return None
        r = _ret(df["close"].values.astype(float), _RET_LOOKBACK)
        if r is not None:
            _spy_cache, _spy_ts = r, time.monotonic()
        return r
    except Exception as e:
        logger.debug(f"[rs] SPY benchmark fetch failed: {e}")
        return None


def is_rs_leader(ticker: str, daily_df=None) -> tuple[bool, dict]:
    """
    Return (is_leader, detail). Fails closed → (False, {...}) on any data gap.

    is_leader = outperforming SPY over 20d AND 20-EMA rising AND above 50-SMA.
    detail carries the measured values so the signal can be tagged + the
    scorecard can track the exemption's realized expectancy.
    """
    detail: dict = {}
    try:
        import pandas as pd  # noqa: F401  (ensures pandas present for ewm)
        if daily_df is None:
            from engine.alpaca_client import get_bars
            daily_df = get_bars(ticker, timeframe="1Day", days=90)
        if daily_df is None or len(daily_df) < 51:
            return False, {"reason": "insufficient daily bars"}

        closes = daily_df["close"].values.astype(float)
        ema20 = daily_df["close"].ewm(span=20, adjust=False).mean().values.astype(float)

        ret20 = _ret(closes, _RET_LOOKBACK)
        spy20 = _spy_ret20()
        if ret20 is None or spy20 is None:
            return False, {"reason": "return data unavailable"}

        ema_rising   = bool(ema20[-1] > ema20[-1 - _EMA_SLOPE_BARS])
        sma50        = float(np.mean(closes[-50:]))
        above_sma50  = bool(float(closes[-1]) > sma50)
        outperforms  = bool(ret20 > spy20)

        detail = {
            "rs_vs_spy_pct": round((ret20 - spy20) * 100, 2),
            "ret20_pct":     round(ret20 * 100, 2),
            "spy_ret20_pct": round(spy20 * 100, 2),
            "ema20_rising":  ema_rising,
            "above_sma50":   above_sma50,
        }
        ok = outperforms and ema_rising and above_sma50
        return ok, detail
    except Exception as e:
        logger.debug(f"[rs] is_rs_leader({ticker}) failed: {e}")
        return False, {"reason": f"error: {e}"}
