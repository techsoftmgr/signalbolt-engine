"""
Chop Detector
=============
Pre-filter that identifies low-quality, non-directional market environments
BEFORE running SMC analysis or scoring. Prevents wasting compute on setups
that form in choppy conditions where SMC patterns have low predictive value.

Returns a ChopResult — when is_choppy=True the caller should:
  - For trend strategies: skip the ticker entirely
  - For mean-reversion: proceed (chop IS the environment)
  - For watchlist: allow entry with penalty

Checks (each contributes a weighted penalty 0–100):
  1. ADX strength          (weight 30) — primary trend filter
  2. VWAP slope            (weight 15) — directional drift
  3. Candle body overlap   (weight 20) — price coiling
  4. ATR compression       (weight 15) — volatility contraction
  5. Directional efficiency(weight 15) — net move / total path
  6. Volume participation  (weight  5) — institutional presence

Score interpretation:
  0–30   → Clean directional environment  → green light
  31–55  → Mixed environment              → caution, reduce size
  56–100 → Choppy                         → skip (trend) / proceed (mean-rev)

Thresholds vary by regime — RANGING regime has higher tolerance since
choppy IS the expected environment for mean-reversion setups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger("signalbolt.chop")

# ── Per-regime chop tolerance thresholds ─────────────────────────────────────
# Higher threshold = more tolerant of chop before blocking.
_CHOP_THRESHOLD_BY_REGIME: dict[str, float] = {
    "TRENDING_BULL": 35.0,
    "TRENDING_BEAR": 35.0,
    "RANGING":       65.0,   # higher tolerance — mean-reversion trades IN chop
    "HIGH_VOL":      45.0,   # volatile but may still have direction
    "LOW_VOL":       40.0,   # low vol + chop = worst environment
    "PANIC":         55.0,   # panic moves can be choppy initially
    "RISK_OFF":      40.0,
}
_DEFAULT_THRESHOLD = 40.0

# ── Per-strategy strictness multiplier ───────────────────────────────────────
# Scalping needs the cleanest environment — apply extra strictness
_STRATEGY_STRICTNESS: dict[str, float] = {
    "scalping":     0.80,  # threshold × 0.80 → stricter gate
    "day_trade":    1.00,
    "swing_trade":  1.10,  # swing can tolerate more intraday chop
    "options_flow": 1.00,
    "dark_pool":    1.00,
}

# ── Chop component thresholds ─────────────────────────────────────────────────
ADX_STRONG_TREND     = 25.0   # ADX > 25 → confirmed trend
ADX_WEAK_TREND       = 18.0   # ADX < 18 → no meaningful trend (Wilder's classic)
VWAP_SLOPE_MIN       = 0.0003 # 0.03% per bar; below = VWAP effectively flat
OVERLAP_CHOP_LEVEL   = 0.65   # >65% bars overlapping prior bar body = coiling
ATR_COMPRESS_RATIO   = 0.70   # current ATR < 70% of long ATR = compression
DE_NOISY_THRESHOLD   = 0.30   # directional efficiency < 30% = noise-dominated
VOL_PARTICIPATION    = 0.80   # volume < 80% of 20-bar avg = thin market


@dataclass
class ChopResult:
    """Result of chop detection analysis."""
    chop_score:             float          # 0–100 (higher = choppier)
    is_choppy:              bool           # True if above regime threshold
    reasons:                list[str]      = field(default_factory=list)
    regime_note:            str            = ""
    adx:                    float          = 0.0
    vol_ratio:              float          = 0.0
    directional_efficiency: float          = 0.0
    threshold_used:         float          = 0.0

    def as_penalty(self) -> float:
        """Score penalty to apply to confluence scorer (0–15 pts max)."""
        if not self.is_choppy:
            return 0.0
        # Scale penalty: 56 chop → 3 pts, 80 chop → 10 pts, 100 chop → 15 pts
        excess = max(0, self.chop_score - self.threshold_used)
        return min(15.0, excess * 0.3)


# ── Internal metric computers ─────────────────────────────────────────────────

def _compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """ADX(14) via Wilder smoothing. Returns 20.0 on insufficient data."""
    try:
        highs  = df["high"].tolist()
        lows   = df["low"].tolist()
        closes = df["close"].tolist()
        n = len(closes)
        if n < period + 2:
            return 20.0

        plus_dm_list, minus_dm_list, tr_list = [], [], []
        for i in range(1, n):
            up   = highs[i]  - highs[i - 1]
            down = lows[i-1] - lows[i]
            plus_dm_list.append(up   if up > down and up > 0 else 0.0)
            minus_dm_list.append(down if down > up and down > 0 else 0.0)
            tr_list.append(max(
                highs[i] - lows[i],
                abs(highs[i]  - closes[i - 1]),
                abs(lows[i]   - closes[i - 1]),
            ))

        def _smooth(lst: list, p: int) -> list:
            if len(lst) < p:
                return [0.0] * len(lst)
            result = [sum(lst[:p]) / p]
            for v in lst[p:]:
                result.append((result[-1] * (p - 1) + v) / p)
            # Pad front so index aligns
            return [0.0] * (p - 1) + result

        atr_s   = _smooth(tr_list,       period)
        plus_s  = _smooth(plus_dm_list,  period)
        minus_s = _smooth(minus_dm_list, period)

        dx_list = []
        for a, p_, m in zip(atr_s, plus_s, minus_s):
            if a == 0:
                continue
            pd_  = (p_ / a) * 100
            md_  = (m  / a) * 100
            di_s = pd_ + md_
            dx_list.append(abs(pd_ - md_) / di_s * 100 if di_s > 0 else 0.0)

        if not dx_list:
            return 20.0
        return float(sum(dx_list[-period:]) / min(period, len(dx_list)))
    except Exception:
        return 20.0


def _vwap_slope_pct_per_bar(df: pd.DataFrame, lookback: int = 10) -> float:
    """VWAP % change per bar over lookback bars. Returns large value if non-flat."""
    try:
        if len(df) < lookback + 1:
            return VWAP_SLOPE_MIN * 2  # assume non-flat if insufficient data
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_tpv = (typical * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        vwap    = cum_tpv / cum_vol
        window  = vwap.iloc[-lookback:]
        if vwap.iloc[-lookback] == 0:
            return VWAP_SLOPE_MIN * 2
        slope = abs(float(window.iloc[-1] - window.iloc[0])) / float(abs(window.iloc[0])) / lookback
        return slope
    except Exception:
        return VWAP_SLOPE_MIN * 2  # conservative: assume non-flat


def _body_overlap_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    """Fraction of recent bars whose candle body overlaps the prior bar's body."""
    try:
        tail = df.tail(lookback + 1).reset_index(drop=True)
        if len(tail) < 2:
            return 0.5
        overlapping = 0
        for i in range(1, len(tail)):
            c_hi = max(float(tail.loc[i, "open"]),   float(tail.loc[i, "close"]))
            c_lo = min(float(tail.loc[i, "open"]),   float(tail.loc[i, "close"]))
            p_hi = max(float(tail.loc[i-1, "open"]), float(tail.loc[i-1, "close"]))
            p_lo = min(float(tail.loc[i-1, "open"]), float(tail.loc[i-1, "close"]))
            if c_lo < p_hi and c_hi > p_lo:  # bodies intersect
                overlapping += 1
        return overlapping / (len(tail) - 1)
    except Exception:
        return 0.5


def _atr_compression_ratio(df: pd.DataFrame, short: int = 7, long: int = 20) -> float:
    """
    Short-period avg H-L range / long-period avg H-L range.
    < 0.70 means ATR is compressing — volatility is contracting.
    """
    try:
        hl = (df["high"] - df["low"]).tolist()
        if len(hl) < long:
            return 1.0
        short_avg = sum(hl[-short:]) / short
        long_avg  = sum(hl[-long:])  / long
        return short_avg / long_avg if long_avg > 0 else 1.0
    except Exception:
        return 1.0


def _directional_efficiency(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Net displacement / total path length over lookback bars.
    1.0 = perfectly directional. 0.0 = pure noise (price goes nowhere net).
    Institutional moves have DE > 0.40. Chop is typically < 0.25.
    """
    try:
        closes = df["close"].tolist()[-lookback:]
        if len(closes) < 2:
            return 0.5
        net_move   = abs(closes[-1] - closes[0])
        total_path = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes)))
        return net_move / total_path if total_path > 0 else 0.5
    except Exception:
        return 0.5


def _volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    """Last completed bar volume / avg of prior lookback bars."""
    try:
        vols = df["volume"].tolist()
        if len(vols) < 3:
            return 1.0
        # Use penultimate bar (last completed) vs prior avg
        avg = sum(vols[-(lookback + 1):-1]) / min(lookback, len(vols) - 1)
        return float(vols[-2]) / avg if avg > 0 else 1.0
    except Exception:
        return 1.0


# ── Public entry point ────────────────────────────────────────────────────────

def detect(
    df:             pd.DataFrame,
    regime_type:    str = "UNKNOWN",
    interval:       str = "15m",
    strategy_type:  str = "day_trade",
) -> ChopResult:
    """
    Run all chop checks and return a ChopResult.

    Args:
        df:            OHLCV DataFrame (pre-filtered to market hours)
        regime_type:   from regime_detector.detect() — sets tolerance threshold
        interval:      bar interval — minor effect on lookback tuning
        strategy_type: scalping uses stricter gates

    Returns:
        ChopResult with chop_score, is_choppy flag, and penalty amount.
        Callers should check is_choppy before running SMC analysis.
    """
    if df is None or len(df) < 20:
        return ChopResult(
            chop_score=50.0,
            is_choppy=False,
            reasons=["Insufficient data for chop check"],
            threshold_used=_DEFAULT_THRESHOLD,
        )

    reasons: list[str] = []
    penalty = 0.0

    # ── 1. ADX — trend strength (weight: 30) ──────────────────────────────────
    adx = _compute_adx(df, period=14)
    if adx < ADX_WEAK_TREND:
        p = min(30.0, (ADX_WEAK_TREND - adx) / ADX_WEAK_TREND * 30.0)
        penalty += p
        reasons.append(f"ADX={adx:.1f} < {ADX_WEAK_TREND} — no directional trend")
    elif adx < ADX_STRONG_TREND:
        penalty += 10.0
        reasons.append(f"ADX={adx:.1f} — weak trend (< {ADX_STRONG_TREND})")

    # ── 2. VWAP slope — directional drift (weight: 15) ────────────────────────
    vslope = _vwap_slope_pct_per_bar(df, lookback=10)
    if vslope < VWAP_SLOPE_MIN:
        p = 15.0 * (1.0 - min(1.0, vslope / VWAP_SLOPE_MIN))
        penalty += p
        reasons.append(f"VWAP flat (slope {vslope*100:.4f}%/bar < {VWAP_SLOPE_MIN*100:.4f}% threshold)")

    # ── 3. Candle body overlap — coiling (weight: 20) ─────────────────────────
    overlap = _body_overlap_ratio(df, lookback=20)
    if overlap > OVERLAP_CHOP_LEVEL:
        p = (overlap - OVERLAP_CHOP_LEVEL) / (1.0 - OVERLAP_CHOP_LEVEL) * 20.0
        penalty += p
        reasons.append(f"Heavy body overlap ({overlap*100:.0f}% of bars coiling in range)")

    # ── 4. ATR compression — volatility contraction (weight: 15) ─────────────
    atr_ratio = _atr_compression_ratio(df, short=7, long=20)
    if atr_ratio < ATR_COMPRESS_RATIO:
        p = (ATR_COMPRESS_RATIO - atr_ratio) / ATR_COMPRESS_RATIO * 15.0
        penalty += p
        reasons.append(f"ATR compressing (7-bar / 20-bar ratio = {atr_ratio:.2f})")

    # ── 5. Directional efficiency — noise vs signal (weight: 15) ──────────────
    de = _directional_efficiency(df, lookback=20)
    if de < DE_NOISY_THRESHOLD:
        p = (DE_NOISY_THRESHOLD - de) / DE_NOISY_THRESHOLD * 15.0
        penalty += p
        reasons.append(f"Low directional efficiency ({de:.3f}) — price going nowhere net")

    # ── 6. Volume participation — institutional presence (weight: 5) ──────────
    vol_ratio = _volume_ratio(df, lookback=20)
    if vol_ratio < VOL_PARTICIPATION:
        penalty += 5.0
        reasons.append(f"Thin volume ({vol_ratio:.2f}× avg) — low institutional participation")

    chop_score = min(100.0, penalty)

    # ── Apply regime threshold ─────────────────────────────────────────────────
    base_threshold  = _CHOP_THRESHOLD_BY_REGIME.get(regime_type, _DEFAULT_THRESHOLD)
    strictness      = _STRATEGY_STRICTNESS.get(strategy_type, 1.0)
    threshold       = base_threshold * strictness
    is_choppy       = chop_score >= threshold

    logger.debug(
        f"[chop] regime={regime_type} strategy={strategy_type} "
        f"score={chop_score:.1f} threshold={threshold:.1f} is_choppy={is_choppy} "
        f"ADX={adx:.1f} DE={de:.3f} overlap={overlap:.2f} vol={vol_ratio:.2f}"
    )

    return ChopResult(
        chop_score=round(chop_score, 1),
        is_choppy=is_choppy,
        reasons=reasons,
        regime_note=f"Regime={regime_type}, threshold={threshold:.0f}, strategy={strategy_type}",
        adx=round(adx, 1),
        vol_ratio=round(vol_ratio, 2),
        directional_efficiency=round(de, 3),
        threshold_used=round(threshold, 1),
    )
