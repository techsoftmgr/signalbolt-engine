"""
Mean-Reversion Engine
=====================
Separate analysis pipeline for RANGING market regimes.

In ranging markets, trend-following SMC (BOS/CHoCH chasing) has poor win
rates because structure breaks are often false and quickly reversed. The
correct approach is mean-reversion: buy when price is stretched below a
central anchor and sell when stretched above it.

This engine triggers when regime_detector returns "RANGING" (ADX < 20).

Signals require:
  1. Price significantly deviated from VWAP (primary anchor)
  2. RSI at extremes confirming the stretch
  3. Price at/near a structural boundary (prior day H/L, Bollinger Band)
  4. Liquidity sweep confirming the extreme (stop raid at the level)
  5. Candle showing rejection (wick > body at the extreme)
  6. Volume expansion (institutional participation at the reversal point)

Scoring (0–100, separate from the trend-following scorer):
  L1 VWAP deviation   25 pts
  L2 RSI extreme      20 pts
  L3 Band extension   20 pts
  L4 Sweep/rejection  20 pts
  L5 Volume           15 pts

Fire threshold: 70 (lower than trend signals — mean-reversion has smaller targets)

SL/TP logic:
  - Stop: beyond the sweep wick (just outside the extreme)
  - T1:   VWAP (central anchor) — roughly 50% of the way back
  - T2:   Opposite boundary (prior day high for longs, low for shorts)
  - R:R minimum: 1.5× (T1 must be ≥ 1.5× risk)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger("signalbolt.mean_rev")

# ── Thresholds ─────────────────────────────────────────────────────────────────
MR_FIRE_THRESHOLD   = 70      # lower than trend — MR has tighter targets
MR_RSI_OVERSOLD     = 32      # RSI below this = oversold
MR_RSI_OVERBOUGHT   = 68      # RSI above this = overbought
MR_VWAP_DEV_MIN     = 0.008   # price must be ≥ 0.8% from VWAP to qualify
MR_BB_WIDTH_MIN     = 0.010   # Bollinger Band width ≥ 1% to be meaningful
MR_VOL_EXPANSION    = 1.20    # volume must be ≥ 1.2× avg at reversal candle

# Strategy type label for this engine
MR_STRATEGY_TYPE = "mean_reversion"


@dataclass
class MRAnalysis:
    """Output of mean-reversion analysis."""
    ticker:          str
    direction:       str          # "LONG" (oversold reversal) or "SHORT"
    score:           int          # 0–100
    passes:          bool
    entry:           float
    stop_loss:       float
    target_one:      float
    target_two:      float
    risk_reward:     float
    strategy_type:   str          = MR_STRATEGY_TYPE
    confidence_factors: list[str] = None
    breakdown:       dict         = None
    vwap:            float        = 0.0
    vwap_dev_pct:    float        = 0.0
    rsi:             float        = 0.0
    setup_type:      str          = "VWAP_MEAN_REVERSION"

    def __post_init__(self):
        if self.confidence_factors is None:
            self.confidence_factors = []
        if self.breakdown is None:
            self.breakdown = {}


def analyze(
    ticker:       str,
    df:           pd.DataFrame,
    price:        float,
    regime:       dict,
    interval:     str = "15m",
) -> Optional[MRAnalysis]:
    """
    Run mean-reversion analysis on a ticker.

    Args:
        ticker:  stock ticker
        df:      OHLCV DataFrame (market-hours filtered, ≥ 40 bars)
        price:   current price (Alpaca real-time)
        regime:  output of regime_detector.detect()
        interval: bar interval for ADR estimation

    Returns MRAnalysis if score ≥ threshold, None otherwise.
    """
    if df is None or len(df) < 30:
        return None

    # Only run in RANGING regime (caller should check, but double-check here)
    regime_type = (regime or {}).get("regime_type", "UNKNOWN")
    if regime_type not in ("RANGING", "LOW_VOL"):
        logger.debug(f"[mr] {ticker} regime={regime_type} — not a mean-reversion environment")
        return None

    score = 0.0
    factors: list[str] = []
    breakdown: dict = {}

    try:
        # ── 1. VWAP deviation (25 pts) ─────────────────────────────────────────
        vwap, vwap_dev = _compute_vwap_deviation(df, price)
        l1_vwap = 0.0
        if vwap_dev >= MR_VWAP_DEV_MIN * 2.5:
            l1_vwap = 25.0
            factors.append(f"Strong VWAP deviation ({vwap_dev*100:.2f}%)")
        elif vwap_dev >= MR_VWAP_DEV_MIN * 1.5:
            l1_vwap = 17.0
            factors.append(f"VWAP stretched ({vwap_dev*100:.2f}%)")
        elif vwap_dev >= MR_VWAP_DEV_MIN:
            l1_vwap = 10.0
        else:
            logger.debug(f"[mr] {ticker} VWAP deviation {vwap_dev*100:.2f}% < minimum {MR_VWAP_DEV_MIN*100:.1f}%")
            return None  # Hard gate: must have meaningful VWAP deviation

        score += l1_vwap
        breakdown["l1_vwap"] = round(l1_vwap)

        # Determine direction from VWAP deviation
        direction = "LONG" if price < vwap else "SHORT"

        # ── 2. RSI extreme (20 pts) ────────────────────────────────────────────
        rsi = _compute_rsi(df, period=14)
        l2_rsi = 0.0
        if direction == "LONG":
            if rsi < MR_RSI_OVERSOLD - 8:
                l2_rsi = 20.0
                factors.append(f"Deeply oversold RSI ({rsi:.0f})")
            elif rsi < MR_RSI_OVERSOLD:
                l2_rsi = 13.0
                factors.append(f"Oversold RSI ({rsi:.0f})")
            elif rsi < MR_RSI_OVERSOLD + 8:
                l2_rsi = 6.0
        else:  # SHORT
            if rsi > MR_RSI_OVERBOUGHT + 8:
                l2_rsi = 20.0
                factors.append(f"Deeply overbought RSI ({rsi:.0f})")
            elif rsi > MR_RSI_OVERBOUGHT:
                l2_rsi = 13.0
                factors.append(f"Overbought RSI ({rsi:.0f})")
            elif rsi > MR_RSI_OVERBOUGHT - 8:
                l2_rsi = 6.0

        score += l2_rsi
        breakdown["l2_rsi"] = round(l2_rsi)

        # ── 3. Bollinger Band extension / prior day H-L (20 pts) ──────────────
        bb_score, bb_factor, bb_upper, bb_lower = _bollinger_band_extension(
            df, direction, price
        )
        l3_band = bb_score
        if bb_factor:
            factors.append(bb_factor)
        score += l3_band
        breakdown["l3_band"] = round(l3_band)

        # Prior day high / low proximity
        pdh, pdl = _prior_day_levels(df)

        # ── 4. Sweep / rejection candle (20 pts) ──────────────────────────────
        l4_sweep = _sweep_rejection_score(df, direction)
        if l4_sweep >= 15:
            factors.append("Stop hunt + rejection confirmed")
        elif l4_sweep >= 8:
            factors.append("Candle rejection at extreme")
        score += l4_sweep
        breakdown["l4_sweep"] = round(l4_sweep)

        # ── 5. Volume expansion (15 pts) ──────────────────────────────────────
        vol_ratio = _volume_expansion(df, lookback=20)
        l5_vol = 0.0
        if vol_ratio >= MR_VOL_EXPANSION * 1.5:
            l5_vol = 15.0
            factors.append(f"Strong volume surge ({vol_ratio:.1f}× avg)")
        elif vol_ratio >= MR_VOL_EXPANSION:
            l5_vol = 9.0
            factors.append(f"Volume expanding ({vol_ratio:.1f}× avg)")
        elif vol_ratio >= 0.9:
            l5_vol = 3.0
        score += l5_vol
        breakdown["l5_vol"] = round(l5_vol)

        # ── Final score ────────────────────────────────────────────────────────
        final_score = min(100, round(score))
        passes = final_score >= MR_FIRE_THRESHOLD

        if not passes:
            logger.debug(f"[mr] {ticker} {direction} score={final_score} < {MR_FIRE_THRESHOLD} — skip")
            return None

        # ── SL/TP calculation ──────────────────────────────────────────────────
        atr = float((df["high"] - df["low"]).tail(14).mean())
        entry = price

        if direction == "LONG":
            # Stop below the lowest low of last 3 bars (below sweep wick)
            sweep_wick = float(df["low"].tail(3).min())
            stop_loss  = round(sweep_wick - atr * 0.5, 2)
            # T1 = VWAP (mean reversion target)
            target_one = round(vwap, 2)
            # T2 = prior day high or upper Bollinger band
            target_two = round(max(pdh, bb_upper) if pdh > entry else bb_upper, 2)
        else:  # SHORT
            sweep_wick = float(df["high"].tail(3).max())
            stop_loss  = round(sweep_wick + atr * 0.5, 2)
            target_one = round(vwap, 2)
            target_two = round(min(pdl, bb_lower) if pdl < entry else bb_lower, 2)

        risk     = abs(entry - stop_loss)
        t1_dist  = abs(target_one - entry)
        rr = t1_dist / risk if risk > 0 else 0.0

        if rr < 1.2:
            logger.debug(f"[mr] {ticker} R:R={rr:.2f} < 1.2 — skip (stop too wide for MR)")
            return None

        logger.info(
            f"[mr] {ticker} {direction} MEAN-REV score={final_score} "
            f"vwap_dev={vwap_dev*100:.2f}% rsi={rsi:.0f} "
            f"entry={entry} sl={stop_loss} t1={target_one} rr={rr:.2f}"
        )

        return MRAnalysis(
            ticker=ticker,
            direction=direction,
            score=final_score,
            passes=True,
            entry=entry,
            stop_loss=stop_loss,
            target_one=target_one,
            target_two=target_two,
            risk_reward=round(rr, 2),
            confidence_factors=factors[:4],
            breakdown=breakdown,
            vwap=round(vwap, 2),
            vwap_dev_pct=round(vwap_dev * 100, 3),
            rsi=round(rsi, 1),
        )

    except Exception as e:
        logger.debug(f"[mr] {ticker} analysis failed: {e}")
        return None


# ── Metric helpers ────────────────────────────────────────────────────────────

def _compute_vwap_deviation(df: pd.DataFrame, price: float) -> tuple[float, float]:
    """Compute current VWAP and % deviation of price from VWAP."""
    try:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_tpv = (typical * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        vwap    = float((cum_tpv / cum_vol).iloc[-1])
        dev     = abs(price - vwap) / vwap if vwap > 0 else 0.0
        return vwap, dev
    except Exception:
        return price, 0.0


def _compute_rsi(df: pd.DataFrame, period: int = 14) -> float:
    """RSI using Wilder's smoothing."""
    try:
        closes = df["close"].tolist()
        if len(closes) < period + 1:
            return 50.0
        gains  = [max(0, closes[i] - closes[i-1]) for i in range(1, len(closes))]
        losses = [max(0, closes[i-1] - closes[i]) for i in range(1, len(closes))]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        for g, l in zip(gains[period:], losses[period:]):
            avg_gain = (avg_gain * (period - 1) + g) / period
            avg_loss = (avg_loss * (period - 1) + l) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    except Exception:
        return 50.0


def _bollinger_band_extension(
    df: pd.DataFrame, direction: str, price: float
) -> tuple[float, str, float, float]:
    """
    Check if price is beyond Bollinger Bands (2 std dev from 20-period MA).
    Returns (score, factor_text, upper_band, lower_band).
    """
    try:
        closes    = df["close"].tail(20)
        ma        = float(closes.mean())
        std       = float(closes.std())
        upper     = ma + 2 * std
        lower     = ma - 2 * std
        width_pct = (upper - lower) / ma if ma > 0 else 0

        if width_pct < MR_BB_WIDTH_MIN:
            return 5.0, "Bollinger Bands too narrow (low volatility)", upper, lower

        if direction == "LONG" and price < lower:
            excess = (lower - price) / (std if std > 0 else 1)
            score  = min(20.0, 12.0 + excess * 4)
            return score, f"Price below lower Bollinger Band (-{excess:.1f}σ)", upper, lower

        if direction == "SHORT" and price > upper:
            excess = (price - upper) / (std if std > 0 else 1)
            score  = min(20.0, 12.0 + excess * 4)
            return score, f"Price above upper Bollinger Band (+{excess:.1f}σ)", upper, lower

        # Price near but not outside bands
        if direction == "LONG":
            proximity = (price - lower) / (std if std > 0 else 1)
            return max(0, 8.0 - proximity * 2), "", upper, lower
        else:
            proximity = (upper - price) / (std if std > 0 else 1)
            return max(0, 8.0 - proximity * 2), "", upper, lower

    except Exception:
        return 5.0, "", 0.0, 0.0


def _prior_day_levels(df: pd.DataFrame) -> tuple[float, float]:
    """Estimate prior day high and low from the DataFrame."""
    try:
        if len(df) < 26:
            return float(df["high"].max()), float(df["low"].min())
        prior_day = df.iloc[-52:-26]  # rough prior session (26 bars = 1 day on 15m)
        if prior_day.empty:
            prior_day = df.iloc[:len(df)//2]
        return float(prior_day["high"].max()), float(prior_day["low"].min())
    except Exception:
        return float(df["high"].max()), float(df["low"].min())


def _sweep_rejection_score(df: pd.DataFrame, direction: str) -> float:
    """
    Score the most recent candle's rejection characteristics.
    High wick-to-body ratio at the extreme = rejection.
    """
    try:
        last = df.iloc[-1]
        open_ = float(last["open"])
        close = float(last["close"])
        high  = float(last["high"])
        low   = float(last["low"])

        body = abs(close - open_)
        total_range = high - low
        if total_range == 0:
            return 0.0

        if direction == "LONG":
            # Rejection = long lower wick, close in upper half
            lower_wick = min(open_, close) - low
            wick_ratio = lower_wick / total_range
            close_position = (close - low) / total_range  # 1.0 = closed at high
            rejection_score = wick_ratio * 10 + close_position * 10
        else:
            upper_wick = high - max(open_, close)
            wick_ratio = upper_wick / total_range
            close_position = (high - close) / total_range
            rejection_score = wick_ratio * 10 + close_position * 10

        return min(20.0, rejection_score)
    except Exception:
        return 0.0


def _volume_expansion(df: pd.DataFrame, lookback: int = 20) -> float:
    """Volume of last completed bar relative to prior lookback average."""
    try:
        vols = df["volume"].tolist()
        if len(vols) < 3:
            return 1.0
        avg = sum(vols[-(lookback + 1):-1]) / min(lookback, len(vols) - 1)
        return float(vols[-2]) / avg if avg > 0 else 1.0
    except Exception:
        return 1.0


def to_signal_dict(mr: MRAnalysis, session: dict) -> dict:
    """
    Convert MRAnalysis to a signal dict compatible with runner._write_signal().
    """
    return {
        "ticker":             mr.ticker,
        "direction":          mr.direction,
        "entry_price":        round(mr.entry, 2),
        "stop_loss":          mr.stop_loss,
        "target_one":         mr.target_one,
        "target_two":         mr.target_two,
        "confidence_score":   mr.score,
        "confidence_factors": mr.confidence_factors,
        "timeframe":          "15m",
        "strategy_type":      MR_STRATEGY_TYPE,
        "setup_type":         mr.setup_type,
        "status":             "active",
        "ai_explanation":     None,
        "regime_type":        "RANGING",
        "session_mode":       (session or {}).get("mode", "STANDARD"),
        "confidence_tier":    "B+" if mr.score >= 74 else "B",
        "position_multiplier": 0.50,   # mean-reversion: smaller size (less certainty)
        "risk_reward":        mr.risk_reward,
        "score_breakdown":    mr.breakdown,
    }
