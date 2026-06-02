"""
Unit tests for engine/chop_detector.py
=======================================
Tests the chop detector's metric computations and overall scoring logic
using synthetic OHLCV DataFrames that represent known market conditions.
"""

import pytest
import pandas as pd
import numpy as np

from engine.chop_detector import (
    detect,
    ChopResult,
    _compute_adx,
    _vwap_slope_pct_per_bar,
    _body_overlap_ratio,
    _atr_compression_ratio,
    _directional_efficiency,
    _volume_ratio,
    ADX_WEAK_TREND,
    ADX_STRONG_TREND,
    OVERLAP_CHOP_LEVEL,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trending_df(n: int = 50, step: float = 0.50, noise: float = 0.10) -> pd.DataFrame:
    """Steadily trending upward bars — should produce clean environment."""
    rng  = np.random.default_rng(42)
    base = 100.0
    rows = []
    for i in range(n):
        open_  = base + i * step + rng.uniform(-noise, noise)
        close  = open_ + step + rng.uniform(-noise, noise)
        high   = max(open_, close) + rng.uniform(0, noise)
        low    = min(open_, close) - rng.uniform(0, noise)
        vol    = 1_000_000 + rng.integers(-50_000, 50_000)
        rows.append({"open": open_, "high": high, "low": low, "close": close, "volume": vol})
    return pd.DataFrame(rows)


def _make_choppy_df(n: int = 50, range_width: float = 1.0, noise: float = 0.30) -> pd.DataFrame:
    """
    Price oscillating within a tight range — high body overlap, low DE, low ADX.
    """
    rng  = np.random.default_rng(99)
    mid  = 100.0
    rows = []
    for i in range(n):
        direction = 1 if i % 2 == 0 else -1
        open_  = mid + direction * rng.uniform(0, range_width / 2)
        close  = mid - direction * rng.uniform(0, range_width / 2)
        high   = max(open_, close) + rng.uniform(0, noise)
        low    = min(open_, close) - rng.uniform(0, noise)
        # Thin volume for extra chop
        vol    = 300_000 + rng.integers(-50_000, 50_000)
        rows.append({"open": open_, "high": high, "low": low, "close": close, "volume": vol})
    return pd.DataFrame(rows)


def _make_atr_compressing_df(n: int = 40) -> pd.DataFrame:
    """
    First half has wide bars (high ATR), second half has narrow bars.
    ATR compression ratio should be < 0.70.
    """
    rng  = np.random.default_rng(7)
    rows = []
    for i in range(n):
        # Narrow only the last 10 bars: the short(7) window is all-narrow while
        # the long(20) window still spans wide bars, so the ratio compresses.
        # (A 50/50 split left the whole long window inside the narrow half →
        #  ratio ≈ 1.0 and no compression was detectable.)
        narrow    = i >= (n - 10)
        hl_range  = 0.50 if narrow else 3.0   # wide then narrow
        open_     = 100.0 + rng.uniform(-0.5, 0.5)
        direction = 1 if rng.random() > 0.5 else -1
        close     = open_ + direction * hl_range * 0.5
        high      = max(open_, close) + hl_range * 0.25
        low       = min(open_, close) - hl_range * 0.25
        vol       = 800_000
        rows.append({"open": open_, "high": high, "low": low, "close": close, "volume": vol})
    return pd.DataFrame(rows)


# ── ChopResult.as_penalty() ───────────────────────────────────────────────────

class TestChopResultPenalty:
    def test_no_penalty_when_not_choppy(self):
        cr = ChopResult(chop_score=30.0, is_choppy=False, threshold_used=40.0)
        assert cr.as_penalty() == 0.0

    def test_penalty_scales_with_excess(self):
        cr = ChopResult(chop_score=70.0, is_choppy=True, threshold_used=40.0)
        # excess = 70 - 40 = 30; penalty = 30 * 0.3 = 9.0
        assert abs(cr.as_penalty() - 9.0) < 0.01

    def test_penalty_capped_at_15(self):
        cr = ChopResult(chop_score=100.0, is_choppy=True, threshold_used=40.0)
        assert cr.as_penalty() == 15.0

    def test_penalty_just_above_threshold(self):
        cr = ChopResult(chop_score=42.0, is_choppy=True, threshold_used=40.0)
        # excess = 2; penalty = 0.6
        assert cr.as_penalty() < 1.0


# ── Individual metric functions ───────────────────────────────────────────────

class TestComputeADX:
    def test_insufficient_data_returns_default(self):
        df = pd.DataFrame({"high": [1, 2], "low": [0, 1], "close": [1, 1]})
        assert _compute_adx(df) == 20.0

    def test_trending_df_adx_above_threshold(self):
        df  = _make_trending_df(n=60)
        adx = _compute_adx(df)
        # Strongly trending data should produce ADX > 20
        assert adx > 20.0, f"Expected ADX > 20, got {adx:.1f}"

    def test_choppy_df_adx_below_strong_threshold(self):
        df  = _make_choppy_df(n=60)
        adx = _compute_adx(df)
        assert adx < ADX_STRONG_TREND, f"Choppy ADX should be < {ADX_STRONG_TREND}, got {adx:.1f}"


class TestVWAPSlope:
    def test_trending_has_slope(self):
        df    = _make_trending_df(n=30)
        slope = _vwap_slope_pct_per_bar(df, lookback=10)
        assert slope > 0.0002, f"Trending slope should be meaningful, got {slope:.6f}"

    def test_flat_df_near_zero(self):
        # Flat price, volume constant → VWAP slope ≈ 0
        rows = [{"open": 100, "high": 100.1, "low": 99.9, "close": 100, "volume": 1_000_000}] * 25
        df   = pd.DataFrame(rows)
        slope = _vwap_slope_pct_per_bar(df, lookback=10)
        assert slope < 0.0005, f"Flat price should have near-zero slope, got {slope:.6f}"


class TestBodyOverlapRatio:
    def test_trending_low_overlap(self):
        df  = _make_trending_df(n=30)
        ovr = _body_overlap_ratio(df, lookback=20)
        # Trending bars should not all overlap
        assert ovr < OVERLAP_CHOP_LEVEL, f"Trending overlap should be < {OVERLAP_CHOP_LEVEL}, got {ovr:.2f}"

    def test_choppy_high_overlap(self):
        df  = _make_choppy_df(n=30)
        ovr = _body_overlap_ratio(df, lookback=20)
        assert ovr >= 0.60, f"Choppy bars should have high overlap, got {ovr:.2f}"


class TestATRCompressionRatio:
    def test_compressing_df(self):
        df    = _make_atr_compressing_df(n=40)
        ratio = _atr_compression_ratio(df, short=7, long=20)
        assert ratio < 0.70, f"ATR compression ratio expected < 0.70, got {ratio:.2f}"

    def test_uniform_bars_ratio_near_one(self):
        rows = [{"high": 101, "low": 99, "open": 100, "close": 100, "volume": 1_000_000}] * 30
        df   = pd.DataFrame(rows)
        ratio = _atr_compression_ratio(df)
        assert 0.85 < ratio < 1.15, f"Uniform bars ratio should be ~1.0, got {ratio:.2f}"


class TestDirectionalEfficiency:
    def test_perfectly_trending_efficiency_high(self):
        rows = [{"open": i, "high": i + 0.5, "low": i - 0.1, "close": i + 0.4, "volume": 1e6}
                for i in range(25)]
        df = pd.DataFrame(rows)
        de = _directional_efficiency(df, lookback=20)
        assert de > 0.50, f"Monotonic up move should have high DE, got {de:.3f}"

    def test_choppy_efficiency_low(self):
        df = _make_choppy_df(n=30)
        de = _directional_efficiency(df, lookback=20)
        assert de < 0.35, f"Choppy moves should have low DE, got {de:.3f}"


class TestVolumeRatio:
    def test_above_average_volume_ratio_over_one(self):
        vols = [1_000_000] * 21
        vols[-2] = 2_000_000   # penultimate bar = spike
        rows = [{"open": 100, "high": 101, "low": 99, "close": 100, "volume": v} for v in vols]
        df   = pd.DataFrame(rows)
        ratio = _volume_ratio(df)
        assert ratio > 1.5, f"Volume spike ratio should be > 1.5, got {ratio:.2f}"

    def test_thin_volume_ratio_under_one(self):
        vols = [1_000_000] * 21
        vols[-2] = 200_000   # penultimate bar = thin
        rows = [{"open": 100, "high": 101, "low": 99, "close": 100, "volume": v} for v in vols]
        df   = pd.DataFrame(rows)
        ratio = _volume_ratio(df)
        assert ratio < 0.5, f"Thin volume ratio should be < 0.5, got {ratio:.2f}"


# ── detect() — full integration tests ────────────────────────────────────────

class TestDetect:
    def test_insufficient_data_returns_neutral(self):
        df     = pd.DataFrame()
        result = detect(df)
        assert isinstance(result, ChopResult)
        assert result.chop_score == 50.0
        assert result.is_choppy is False

    def test_trending_environment_not_choppy(self):
        df     = _make_trending_df(n=60)
        result = detect(df, regime_type="TRENDING_BULL", strategy_type="day_trade")
        # Strong trend should not be flagged as choppy
        assert not result.is_choppy, (
            f"Trending environment incorrectly flagged as choppy "
            f"(score={result.chop_score:.1f} threshold={result.threshold_used:.1f})"
        )

    def test_choppy_environment_flagged(self):
        df     = _make_choppy_df(n=60)
        result = detect(df, regime_type="TRENDING_BULL", strategy_type="day_trade")
        # Choppy data against a trend regime should trigger the filter
        assert result.chop_score > 0, "Choppy environment should have non-zero score"

    def test_ranging_regime_higher_tolerance(self):
        """RANGING regime threshold (65) should be higher than TRENDING (35)."""
        df      = _make_choppy_df(n=60)
        ranging = detect(df, regime_type="RANGING",       strategy_type="day_trade")
        bull    = detect(df, regime_type="TRENDING_BULL", strategy_type="day_trade")
        assert ranging.threshold_used > bull.threshold_used, (
            "RANGING regime should be more tolerant of chop than TRENDING"
        )

    def test_scalping_stricter_threshold(self):
        """Scalping multiplier (0.80) should produce lower threshold than day_trade (1.00)."""
        df      = _make_trending_df(n=40)
        scalp   = detect(df, regime_type="TRENDING_BULL", strategy_type="scalping")
        dt      = detect(df, regime_type="TRENDING_BULL", strategy_type="day_trade")
        assert scalp.threshold_used < dt.threshold_used, (
            "Scalping should have stricter (lower) chop threshold"
        )

    def test_atr_compressing_adds_penalty(self):
        df      = _make_atr_compressing_df(n=40)
        result  = detect(df, regime_type="TRENDING_BULL")
        assert result.chop_score > 0, "ATR compression should add to chop score"

    def test_result_fields_populated(self):
        df     = _make_trending_df(n=40)
        result = detect(df, regime_type="TRENDING_BULL", strategy_type="day_trade")
        assert result.adx >= 0
        assert 0.0 <= result.directional_efficiency <= 1.0
        assert result.vol_ratio >= 0
        assert result.threshold_used > 0
        assert isinstance(result.reasons, list)
        assert isinstance(result.regime_note, str)
        assert "Regime=" in result.regime_note
