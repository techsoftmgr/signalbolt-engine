"""
Unit tests — engine/regime_detector.py

Real API:
  detect() -> {
    regime_type, vix, vix_change_pct, adx, above_200ma,
    spy_price, ma200, fear_greed, blocked, block_reason
  }

  _classify(vix, vix_change_pct, above_200ma, adx) -> str
  score_for_signal(regime, direction) -> float  (0-100)

Strategy blocking is indicated by `blocked: True` (PANIC) or via
runner.py which reads `regime_type`. Tests use _classify() for pure
logic and detect() with mocked internal fetchers.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch
import pytest
from engine.regime_detector import (
    _classify,
    score_for_signal,
    VIX_PANIC,
    VIX_HIGH,
    VIX_LOW,
    VIX_SPIKE,
    ADX_TRENDING,
    ADX_RANGING,
)


# ──────────────────────────────────────────────────────────────
# _classify — pure logic, no external calls
# ──────────────────────────────────────────────────────────────

class TestClassifyPure:
    """Tests against the pure _classify() function — no yfinance, no mocks."""

    def test_panic_on_vix_above_30(self):
        assert _classify(35.0, 0.0, True, 20.0) == "PANIC"

    def test_panic_on_vix_exactly_30(self):
        assert _classify(30.1, 0.0, True, 20.0) == "PANIC"

    def test_panic_on_vix_spike(self):
        """VIX 28 but spiked 12% intraday → PANIC."""
        assert _classify(28.0, 0.12, True, 20.0) == "PANIC"

    def test_high_vol_between_25_and_30(self):
        assert _classify(27.0, 0.0, True, 20.0) == "HIGH_VOL"

    def test_high_vol_at_25_01(self):
        assert _classify(25.1, 0.0, True, 20.0) == "HIGH_VOL"

    def test_low_vol_below_15(self):
        assert _classify(12.0, 0.0, True, 20.0) == "LOW_VOL"

    def test_low_vol_at_14_99(self):
        assert _classify(14.99, 0.0, True, 20.0) == "LOW_VOL"

    def test_trending_bull_spy_above_200ma_high_adx(self):
        assert _classify(18.0, 0.0, True, 30.0) == "TRENDING_BULL"

    def test_trending_bear_spy_below_200ma_high_adx(self):
        assert _classify(18.0, 0.0, False, 30.0) == "TRENDING_BEAR"

    def test_ranging_low_adx(self):
        assert _classify(18.0, 0.0, True, 12.0) == "RANGING"

    def test_ranging_at_adx_boundary(self):
        assert _classify(18.0, 0.0, True, ADX_TRENDING - 0.1) == "RANGING"

    def test_panic_overrides_trending(self):
        """Even with high ADX, PANIC regime wins if VIX > 30."""
        assert _classify(35.0, 0.0, True, 40.0) == "PANIC"

    def test_normal_spike_below_threshold(self):
        """VIX spike of 5% (below 10% threshold) should NOT trigger PANIC."""
        result = _classify(20.0, 0.05, True, 20.0)
        assert result != "PANIC"


# ──────────────────────────────────────────────────────────────
# detect() — mocking internal fetchers
# ──────────────────────────────────────────────────────────────

def _run_detect(vix, prev_close=None, above_200ma=True, adx=20.0):
    """Run detect() with mocked _fetch_vix, _fetch_spy_vs_200ma, _fetch_risk_off_signal."""
    if prev_close is None:
        prev_close = vix

    vix_data = {"vix": vix, "prev_close": prev_close}
    spy_data  = {
        "above_200ma": above_200ma,
        "spy_price": 500.0,
        "ma200": 480.0 if above_200ma else 520.0,
        "adx": adx,
        "_hist": None,
    }

    from engine.regime_detector import detect
    with patch("engine.regime_detector._fetch_vix", return_value=vix_data), \
         patch("engine.regime_detector._fetch_spy_vs_200ma", return_value=spy_data), \
         patch("engine.regime_detector._fetch_risk_off_signal", return_value=False):
        return detect()


class TestDetectVIXRegimes:

    def test_panic_regime(self):
        result = _run_detect(vix=35.0)
        assert result["regime_type"] == "PANIC"

    def test_panic_sets_blocked_true(self):
        result = _run_detect(vix=35.0)
        assert result["blocked"] is True

    def test_panic_has_block_reason(self):
        result = _run_detect(vix=35.0)
        assert len(result["block_reason"]) > 0

    def test_panic_on_spike(self):
        result = _run_detect(vix=28.0, prev_close=24.0)  # 16.7% spike
        assert result["regime_type"] == "PANIC"

    def test_high_vol(self):
        result = _run_detect(vix=27.0)
        assert result["regime_type"] == "HIGH_VOL"

    def test_low_vol(self):
        result = _run_detect(vix=12.0)
        assert result["regime_type"] == "LOW_VOL"

    def test_normal_not_blocked(self):
        result = _run_detect(vix=18.0)
        assert result["blocked"] is False

    def test_trending_bull(self):
        result = _run_detect(vix=18.0, above_200ma=True, adx=30.0)
        assert result["regime_type"] == "TRENDING_BULL"

    def test_trending_bear(self):
        result = _run_detect(vix=18.0, above_200ma=False, adx=30.0)
        assert result["regime_type"] == "TRENDING_BEAR"

    def test_ranging(self):
        result = _run_detect(vix=18.0, adx=12.0)
        assert result["regime_type"] == "RANGING"


class TestDetectResultStructure:

    def test_required_keys_present(self):
        result = _run_detect(vix=18.0)
        for key in ["regime_type", "vix", "blocked", "block_reason"]:
            assert key in result, f"Missing key: {key}"

    def test_regime_type_is_valid_string(self):
        result = _run_detect(vix=18.0)
        valid = {"PANIC", "HIGH_VOL", "LOW_VOL", "TRENDING_BULL",
                 "TRENDING_BEAR", "RANGING", "RISK_OFF"}
        assert result["regime_type"] in valid

    def test_vix_value_returned(self):
        result = _run_detect(vix=22.5)
        assert abs(result["vix"] - 22.5) < 0.1

    def test_blocked_is_bool(self):
        result = _run_detect(vix=18.0)
        assert isinstance(result["blocked"], bool)


# ──────────────────────────────────────────────────────────────
# score_for_signal — L6 bonus scoring
# ──────────────────────────────────────────────────────────────

class TestScoreForSignal:

    def _regime(self, regime_type, vix=18.0):
        return {"regime_type": regime_type, "vix": vix, "vix_change_pct": 0.0}

    def test_trending_bull_long_scores_high(self):
        score = score_for_signal(self._regime("TRENDING_BULL"), "LONG")
        assert score >= 80, f"TRENDING_BULL LONG should score >= 80, got {score}"

    def test_trending_bull_short_scores_lower(self):
        bull_long  = score_for_signal(self._regime("TRENDING_BULL"), "LONG")
        bull_short = score_for_signal(self._regime("TRENDING_BULL"), "SHORT")
        assert bull_long > bull_short, "LONG should score higher in bull market"

    def test_trending_bear_short_scores_high(self):
        score = score_for_signal(self._regime("TRENDING_BEAR"), "SHORT")
        assert score >= 80, f"TRENDING_BEAR SHORT should score >= 80, got {score}"

    def test_panic_long_scores_zero(self):
        score = score_for_signal(self._regime("PANIC"), "LONG")
        assert score == 0, f"PANIC LONG should score 0, got {score}"

    def test_score_in_range(self):
        for regime in ["TRENDING_BULL", "TRENDING_BEAR", "RANGING", "HIGH_VOL", "LOW_VOL", "PANIC"]:
            for direction in ["LONG", "SHORT"]:
                score = score_for_signal(self._regime(regime), direction)
                assert 0 <= score <= 100, f"{regime} {direction} score {score} out of range"
