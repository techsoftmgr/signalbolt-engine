"""
Cycle context — the QUALITY differentiators layered on turnaround/peak signals
(memory: signalbolt-cycle-detector). The stuff competitors' oversold/overbought
scanners never answer:

  • cyclicality    — does this name reliably swing/mean-revert? (rank what's even
                     worth watching for the cycle; smooth trenders score low)
  • driver         — is the move MARKET-driven (beta) or COMPANY-specific? The
                     real falling-knife / bull-trap separator.
  • expectedPain   — roughly how much adverse move to expect before it works
                     (ATR-based v1; refined by the track-record MAE once data
                     accrues).

All from daily bars; driver needs a SPY series (already fetched for the
dashboard). Cheap — only computed for the handful of staged turnaround/peak
names per scan.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("signalbolt.cycle_context")

_DRIVER_WINDOW = 40   # bars over which to attribute the move to market vs idiosyncratic


def _cyclicality(df) -> dict:
    """
    How reliably this name oscillates around its own trend. A choppy
    mean-reverter crosses its 50-day MA often with large amplitude (high score);
    a smooth one-way trender rarely crosses (low score).
    """
    out = {"cyclicalityScore": None, "swingAmplitudePct": None}
    try:
        closes = df["close"].astype(float)
        if len(closes) < 60:
            return out
        sma = closes.rolling(50).mean()
        rel = ((closes - sma) / sma).dropna()
        if len(rel) < 20:
            return out
        signs = np.sign(rel.values)
        signs[signs == 0] = 1
        crossings = int(np.sum(np.diff(signs) != 0))   # oscillations around the trend
        amp = float(np.nanmean(np.abs(rel.values)) * 100)   # avg % distance from trend
        score = float(np.clip(crossings * 3.0 + amp * 2.5, 0, 100))
        out["cyclicalityScore"]  = round(score)
        out["swingAmplitudePct"] = round(amp, 1)
    except Exception:
        pass
    return out


def _driver(df, spy_df, window: int = _DRIVER_WINDOW) -> dict:
    """
    Decompose the recent move into a MARKET component (beta × SPY move) and the
    idiosyncratic remainder. marketDrivenPct high => 'it's the market' (more
    likely a buyable dip / squeezable top); low => company-specific (trap risk).
    """
    out = {"beta": None, "marketDrivenPct": None, "driverLabel": None}
    try:
        if spy_df is None or len(df) <= window or len(spy_df) <= window:
            return out
        s = df["close"].astype(float).pct_change().dropna().tail(window).values
        m = spy_df["close"].astype(float).pct_change().dropna().tail(window).values
        n = min(len(s), len(m))
        if n < 10:
            return out
        s, m = s[-n:], m[-n:]
        var_m = float(np.var(m))
        beta = float(np.cov(s, m)[0, 1] / var_m) if var_m > 0 else 1.0
        total = float(df["close"].iloc[-1] / df["close"].iloc[-window] - 1)
        spy_total = float(spy_df["close"].iloc[-1] / spy_df["close"].iloc[-window] - 1)
        market_comp = beta * spy_total
        if abs(total) < 1e-6:
            return out
        market_pct = float(np.clip(abs(market_comp) / abs(total) * 100, 0, 100))
        label = ("market-driven" if market_pct >= 60
                 else "company-specific" if market_pct <= 30
                 else "mixed")
        out["beta"]            = round(beta, 2)
        out["marketDrivenPct"] = round(market_pct)
        out["driverLabel"]     = label
    except Exception:
        pass
    return out


def _expected_pain(df) -> dict:
    """
    Rough adverse-excursion-to-endure before the trade works. v1 = ~2× ATR%
    (a typical swing's chop). Refined by the real track-record MAE as episodes
    accrue. Honest framing: "expect ~X% wobble; it's volatile at the turn."
    """
    out = {"expectedPainPct": None}
    try:
        d = df.sort_index()
        h, l, c = d["high"].astype(float), d["low"].astype(float), d["close"].astype(float)
        pc = c.shift(1)
        import pandas as pd
        tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr = float(tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().iloc[-1])
        last = float(c.iloc[-1])
        if last > 0 and atr > 0:
            atr_pct = atr / last * 100
            out["expectedPainPct"] = round(2.0 * atr_pct, 1)
    except Exception:
        pass
    return out


def compute(df, spy_df=None) -> dict:
    """All cycle-context differentiators for one name. Safe/None on bad data."""
    if df is None or len(df) < 60:
        return {}
    out: dict = {}
    out.update(_cyclicality(df))
    out.update(_driver(df, spy_df))
    out.update(_expected_pain(df))
    return out
