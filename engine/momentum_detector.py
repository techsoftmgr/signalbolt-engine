"""
Systematic Momentum / Trend-Following model
===========================================
The one empirically-proven, out-of-sample-robust effect accessible at our
scale (the AQR / managed-futures approach), as opposed to discretionary chart
patterns. Two ingredients, both required:

  1. TREND filter (time-series momentum): price above SMA50, SMA50 above SMA200
     for longs (mirror for shorts). Only trade names already in a confirmed
     trend — never counter-trend.
  2. CROSS-SECTIONAL momentum: rank the universe by VOLATILITY-ADJUSTED blended
     trailing return (1/3/6-month proxy = 21/63/126 trading days). Trade only
     the strongest (longs) / weakest (shorts) — relative strength, not raw move.

Vol-adjustment (return ÷ annualized vol) stops us chasing high-vol junk that
posts a big number on noise. Ranking is done by the caller across the universe;
this module scores a single name's daily bars.

Holds as a swing (days–weeks) and rides on the swing trailing stop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("signalbolt.momentum")

# ── Tunables ────────────────────────────────────────────────────────────────
_LOOKBACKS      = (21, 63, 126)   # 1 / 3 / 6 month (trading days)
_MIN_BARS       = 150             # need enough history for a trend read
_SMA_FAST       = 50
_SMA_SLOW       = 200             # falls back to 100 if <200 bars

# ── Chase-guard (entry timing) ────────────────────────────────────────────────
# The cross-sectional rank can pick a top-decile name that's a poor ENTRY today:
# either stretched into a blow-off, or already rolling over short-term (TXN fired
# LONG at 314 as it printed a lower high and immediately bled -2%). We gate the
# entry to a "sweet spot" around a short EMA: a healthy pullback that's still
# holding, not an extended spike and not a fresh breakdown. ext_atr = (close −
# EMA) / ATR; the scan skips fires outside [_CHASE_MIN_ATR, _CHASE_MAX_ATR].
_CHASE_EMA      = 10              # short EMA the entry band is measured against
_CHASE_MAX_ATR  = 4.0             # > this ATRs above EMA → chasing a blow-off
_CHASE_MIN_ATR  = -0.5            # < this ATRs below EMA → rolling over (knife)
_ATR_PERIOD     = 14
_MIN_VOL_ADJ    = 0.5             # minimum vol-adjusted momentum to qualify

# A stable, liquid universe for cross-sectional ranking. Momentum needs a
# CONSISTENT universe (not the daily movers list) so relative strength is
# comparable day to day.
UNIVERSE = [
    # mega-cap tech / semis
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "NFLX", "AMD",
    "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ARM", "SMCI",
    # software / cloud / growth
    "CRM", "NOW", "SNOW", "PLTR", "DDOG", "NET", "CRWD", "ZS", "PANW", "FTNT",
    "SHOP", "MELI", "SE", "U", "RBLX", "APP", "TTD", "RDDT", "SPOT",
    # fintech / consumer growth
    "COIN", "HOOD", "SOFI", "AFRM", "UPST", "CELH", "DUOL", "ONON", "DECK",
    "ELF", "HIMS", "SNAP", "PINS", "ABNB", "UBER", "DASH",
    # financials / industrials / energy / healthcare
    "JPM", "GS", "MS", "BAC", "V", "MA", "XOM", "CVX", "COP", "CAT", "DE",
    "BA", "GE", "LLY", "UNH", "MRK", "ABBV", "NVO",
    # consumer / other megacaps
    "COST", "WMT", "HD", "MCD", "NKE", "DIS", "KO", "PEP",
    # crypto-adjacent / high-beta
    "MSTR", "MARA", "RIOT", "CLSK",
    # sector / index ETFs (trend reference)
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "SMH",
]


@dataclass
class MomentumScore:
    ticker:        str
    bias:          str        # 'LONG' / 'SHORT' / 'NONE'
    score:         float      # vol-adjusted blended momentum (signed)
    last_price:    float
    atr:           float
    raw_return:    float      # blended trailing return (fraction)
    ann_vol:       float
    sma_fast:      float
    sma_slow:      float
    ext_atr:       float = 0.0  # signed ATRs of last close vs the short EMA (chase-guard)


def _atr(df: pd.DataFrame, period: int = _ATR_PERIOD) -> float:
    hi = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    cl = df["close"].values.astype(float)
    tr = np.maximum(hi[1:] - lo[1:],
                    np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])))
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) else 0.0
    return float(np.mean(tr[-period:]))


def score(ticker: str, df_daily: pd.DataFrame) -> Optional[MomentumScore]:
    """Score one name's daily bars. Returns None if insufficient data or no
    trend/momentum qualification."""
    if df_daily is None or len(df_daily) < _MIN_BARS:
        return None
    closes = df_daily["close"].values.astype(float)
    last   = float(closes[-1])
    if last <= 0:
        return None

    # Blended trailing return across lookbacks (skips the most recent bar to
    # avoid the 1-bar reversal noise — classic 12-1 style, scaled down).
    rets = []
    for lb in _LOOKBACKS:
        if len(closes) > lb + 1:
            rets.append((closes[-2] - closes[-2 - lb]) / closes[-2 - lb])
    if not rets:
        return None
    raw_return = float(np.mean(rets))

    # Annualized vol from daily log-ish returns over ~63d
    dr = np.diff(closes[-64:]) / closes[-64:-1]
    ann_vol = float(np.std(dr) * np.sqrt(252)) if len(dr) > 5 else 0.0
    if ann_vol <= 0:
        return None
    vol_adj = raw_return / ann_vol      # signed, vol-normalized momentum

    sma_fast = float(np.mean(closes[-_SMA_FAST:]))
    slow_n   = _SMA_SLOW if len(closes) >= _SMA_SLOW else 100
    sma_slow = float(np.mean(closes[-slow_n:]))

    up_trend   = last > sma_fast and sma_fast > sma_slow
    down_trend = last < sma_fast and sma_fast < sma_slow

    bias = "NONE"
    if up_trend and vol_adj >= _MIN_VOL_ADJ:
        bias = "LONG"
    elif down_trend and vol_adj <= -_MIN_VOL_ADJ:
        bias = "SHORT"

    # Chase-guard input: latest close's distance (in ATRs) from a short EMA.
    atr_val   = _atr(df_daily)
    ema_short = float(pd.Series(closes).ewm(span=_CHASE_EMA, adjust=False).mean().iloc[-1])
    ext_atr   = round((last - ema_short) / atr_val, 2) if atr_val > 0 else 0.0

    return MomentumScore(
        ticker=ticker, bias=bias, score=round(vol_adj, 4), last_price=last,
        atr=round(atr_val, 4), raw_return=round(raw_return, 4),
        ann_vol=round(ann_vol, 4), sma_fast=round(sma_fast, 2),
        sma_slow=round(sma_slow, 2), ext_atr=ext_atr,
    )
