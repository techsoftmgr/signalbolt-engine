"""
EMA Reclaim / Trend-Day Continuation Detector
=============================================
Catches the high-momentum trend-day setup (HOOD/CRWD 2026-05-28): after a
flush, price reclaims VWAP and the 9 EMA, the EMAs stack bullishly (9 > 20),
RSI thrusts up out of oversold on a volume surge — then *pulls back to the
9 EMA and holds*. Entry is that first-pullback-hold (the "2nd green bar"),
not the initial thrust.

Paired with a 20-EMA trailing exit in signal_monitor so the winner rides the
whole trend instead of getting cut at a fixed target.

Honest limits:
  - Designed for TRENDING regimes. On chop it whipsaws — must be regime-gated
    upstream (runner._market_allows) and is liquidity-gated by entry_gate.
  - Fires on 15m bar close (the timeframe the setup lives on); not per-tick.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("signalbolt.ema_reclaim")

# ── Tunables ────────────────────────────────────────────────────────────────
_EMA_FAST          = 9
_EMA_SLOW          = 20
_STACK_FLIP_LOOKBACK = 6      # EMA9 must have crossed above EMA20 within N bars
_RSI_THRUST_HI     = 55.0     # RSI now must be above this
_RSI_THRUST_LO     = 42.0     # ...and have been below this within the thrust window
_RSI_THRUST_BARS   = 6
_VOL_SURGE_MULT    = 1.5      # reclaim-leg volume vs 20-bar avg
_VWAP_BARS         = 26       # ~1 RTH session of 15m bars for rolling VWAP
_PULLBACK_TAG_ATR  = 0.5      # pullback low must come within this × ATR of EMA9
_MIN_BARS          = 26


@dataclass
class EMAReclaimSetup:
    direction:   str       # 'LONG' / 'SHORT'
    entry:       float     # current price
    ema9:        float
    ema20:       float
    stop_ref:    float     # pullback extreme (for level-based stop)
    atr:         float
    rsi:         float
    setup_type:  str = "EMA_RECLAIM"


def _atr_hl(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period:
        return 0.0
    hl = (df["high"].values[-period:].astype(float) -
          df["low"].values[-period:].astype(float))
    return float(np.mean(hl))


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI series (same length as closes; first `period` are NaN-ish)."""
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    out = np.full(len(closes), 50.0)
    if len(closes) <= period:
        return out
    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    for i in range(period, len(closes)):
        g = gain[i - 1]; l = loss[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 999.0
        out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def _rolling_vwap(df: pd.DataFrame, bars: int) -> float:
    tail = df.tail(bars)
    tp  = (tail["high"].astype(float) + tail["low"].astype(float) + tail["close"].astype(float)) / 3.0
    vol = tail["volume"].astype(float)
    denom = float(vol.sum())
    if denom <= 0:
        return float(tail["close"].iloc[-1])
    return float((tp * vol).sum() / denom)


def detect(df: pd.DataFrame, current_price: float) -> Optional[EMAReclaimSetup]:
    """Return an EMAReclaimSetup if the trend-reclaim + first-pullback-hold
    pattern completed on the last closed bar, else None."""
    if df is None or len(df) < _MIN_BARS or not current_price:
        return None
    atr = _atr_hl(df, 14)
    if atr <= 0:
        return None

    closes = df["close"].values.astype(float)
    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    vols   = df["volume"].values.astype(float)

    ema9_s  = pd.Series(closes).ewm(span=_EMA_FAST, adjust=False).mean().values
    ema20_s = pd.Series(closes).ewm(span=_EMA_SLOW, adjust=False).mean().values
    rsi_s   = _rsi(closes, 14)
    vwap    = _rolling_vwap(df, _VWAP_BARS)

    ema9, ema20 = float(ema9_s[-1]), float(ema20_s[-1])
    rsi_now     = float(rsi_s[-1])
    vol_avg     = float(np.mean(vols[-20:]))
    vol_surge   = vol_avg > 0 and float(np.max(vols[-5:])) >= _VOL_SURGE_MULT * vol_avg

    # ── LONG ──────────────────────────────────────────────────────────────
    stack_long  = ema9 > ema20
    flip_long   = any(ema9_s[-k] <= ema20_s[-k] for k in range(2, _STACK_FLIP_LOOKBACK + 1))
    rsi_thrust_long = rsi_now >= _RSI_THRUST_HI and \
        float(np.min(rsi_s[-_RSI_THRUST_BARS:])) <= _RSI_THRUST_LO
    above_vwap  = current_price > vwap
    # First-pullback-hold: a recent bar tagged the 9 EMA from above, and the
    # last completed bar closed back above it as a green (reclaim) bar.
    pulled_back = any(lows[-k] <= ema9_s[-k] + _PULLBACK_TAG_ATR * atr for k in range(2, 4))
    last_green_above = closes[-1] > opens[-1] and closes[-1] > ema9
    if (stack_long and flip_long and rsi_thrust_long and above_vwap
            and vol_surge and pulled_back and last_green_above
            and current_price >= ema9):
        stop_ref = float(np.min(lows[-3:]))   # pullback low
        return EMAReclaimSetup(direction="LONG", entry=current_price, ema9=ema9,
                               ema20=ema20, stop_ref=stop_ref, atr=atr, rsi=rsi_now)

    # ── SHORT (mirror) ──────────────────────────────────────────────────────
    stack_short  = ema9 < ema20
    flip_short   = any(ema9_s[-k] >= ema20_s[-k] for k in range(2, _STACK_FLIP_LOOKBACK + 1))
    rsi_thrust_short = rsi_now <= (100 - _RSI_THRUST_HI) and \
        float(np.max(rsi_s[-_RSI_THRUST_BARS:])) >= (100 - _RSI_THRUST_LO)
    below_vwap   = current_price < vwap
    pulled_back_s = any(highs[-k] >= ema9_s[-k] - _PULLBACK_TAG_ATR * atr for k in range(2, 4))
    last_red_below = closes[-1] < opens[-1] and closes[-1] < ema9
    if (stack_short and flip_short and rsi_thrust_short and below_vwap
            and vol_surge and pulled_back_s and last_red_below
            and current_price <= ema9):
        stop_ref = float(np.max(highs[-3:]))
        return EMAReclaimSetup(direction="SHORT", entry=current_price, ema9=ema9,
                               ema20=ema20, stop_ref=stop_ref, atr=atr, rsi=rsi_now)

    return None
