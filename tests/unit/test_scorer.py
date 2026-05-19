"""
Unit tests — engine/scorer.py

Real API:
  score(analysis: dict, strategy_type: str, regime=None, session=None,
        gamma=None, manipulation=None) -> dict
    Returns: total, passes, breakdown, direction, entry, stop_loss, target_one, target_two

  _classify(vix, vix_change_pct, above_200ma, adx) — tested in regime tests

Covers:
  - Strategy weights sum to 100 per strategy
  - Strategy-specific thresholds defined and in valid range
  - Pure logic: _l1_smc scoring with known inputs
  - score() returns required keys
  - Low L1 score blocks signal (L1 hard gate)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from engine.scorer import (
    STRATEGY_WEIGHTS,
    STRATEGY_THRESHOLDS,
    _l1_smc,
)


# ──────────────────────────────────────────────────────────────
# Configuration validation
# ──────────────────────────────────────────────────────────────

class TestScorerConfig:

    def test_all_strategies_have_weights(self):
        for strategy in ["scalping", "day_trade", "swing_trade", "options_flow", "dark_pool"]:
            assert strategy in STRATEGY_WEIGHTS, f"Missing weights for {strategy}"

    def test_all_strategies_have_thresholds(self):
        for strategy in ["scalping", "day_trade", "swing_trade", "options_flow", "dark_pool"]:
            assert strategy in STRATEGY_THRESHOLDS, f"Missing threshold for {strategy}"

    def test_thresholds_in_valid_range(self):
        for strategy, threshold in STRATEGY_THRESHOLDS.items():
            assert 0 < threshold <= 100, f"{strategy} threshold {threshold} out of range"

    def test_strategy_weights_sum_to_100(self):
        for strategy, weights in STRATEGY_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 100) < 0.01, (
                f"{strategy} weights sum to {total}, expected 100"
            )

    def test_weight_values_non_negative(self):
        for strategy, weights in STRATEGY_WEIGHTS.items():
            for layer, w in weights.items():
                assert w >= 0, f"{strategy}.{layer} weight is negative: {w}"

    def test_scalping_heavy_technical_weight(self):
        """Scalping is purely technical — technical weight should be the largest."""
        scalp_w = STRATEGY_WEIGHTS["scalping"]
        max_layer = max(scalp_w, key=scalp_w.get)
        assert max_layer == "technical", f"Scalping's largest weight should be 'technical', got '{max_layer}'"

    def test_swing_heavy_smc_weight(self):
        """Swing trade relies on SMC structure above all else."""
        swing_w = STRATEGY_WEIGHTS["swing_trade"]
        assert swing_w["smc"] >= 30, "Swing SMC weight should be >= 30"

    def test_options_flow_heavy_sentiment(self):
        """Options flow and dark pool are sentiment-driven."""
        flow_w = STRATEGY_WEIGHTS["options_flow"]
        assert flow_w["sentiment"] >= 40, "Options flow sentiment weight should be >= 40"

    def test_options_flow_highest_sentiment(self):
        flow_w = STRATEGY_WEIGHTS["options_flow"]
        dp_w   = STRATEGY_WEIGHTS["dark_pool"]
        scalp_w = STRATEGY_WEIGHTS["scalping"]
        assert flow_w["sentiment"] >= scalp_w["sentiment"]
        assert dp_w["sentiment"]   >= scalp_w["sentiment"]


# ──────────────────────────────────────────────────────────────
# L1 SMC scoring — pure logic
# ──────────────────────────────────────────────────────────────

class TestL1SMC:

    def test_choch_bullish_scores_higher_than_bos(self):
        """CHoCH (character change) is a stronger signal than BOS."""
        choch_score = _l1_smc(
            structure={"choch_bullish": True},
            fvgs={}, obs={}, direction="LONG", price=150.0,
        )
        bos_score = _l1_smc(
            structure={"bos_bullish": True},
            fvgs={}, obs={}, direction="LONG", price=150.0,
        )
        assert choch_score > bos_score

    def test_no_structure_scores_zero_base(self):
        """No structure signals at all → base = 0 (before FVG/OB bonuses)."""
        score = _l1_smc(
            structure={}, fvgs={}, obs={}, direction="LONG", price=150.0
        )
        assert score == 0

    def test_fvg_in_range_adds_score(self):
        """Price inside FVG zone should add points."""
        fvg = {"fvg_bullish": {"top": 151.0, "bottom": 149.0}}
        score_with_fvg = _l1_smc(
            structure={"bos_bullish": True},
            fvgs=fvg, obs={}, direction="LONG", price=150.0,
        )
        score_without_fvg = _l1_smc(
            structure={"bos_bullish": True},
            fvgs={}, obs={}, direction="LONG", price=150.0,
        )
        assert score_with_fvg > score_without_fvg

    def test_ob_containing_price_adds_max_ob_points(self):
        """Price inside order block → maximum OB score contribution."""
        ob = {"ob_bullish": {"top": 151.0, "bottom": 149.0}}
        score_inside = _l1_smc(
            structure={"bos_bullish": True},
            fvgs={}, obs=ob, direction="LONG", price=150.0,  # inside OB
        )
        ob_far = {"ob_bullish": {"top": 200.0, "bottom": 195.0}}
        score_outside = _l1_smc(
            structure={"bos_bullish": True},
            fvgs={}, obs=ob_far, direction="LONG", price=150.0,  # far from OB
        )
        assert score_inside > score_outside

    def test_long_uses_bullish_signals_only(self):
        """LONG direction should not score from bearish CHoCH."""
        long_score = _l1_smc(
            structure={"choch_bearish": True, "choch_bullish": False},
            fvgs={}, obs={}, direction="LONG", price=150.0,
        )
        assert long_score == 0

    def test_short_uses_bearish_signals_only(self):
        """SHORT direction should not score from bullish CHoCH."""
        short_score = _l1_smc(
            structure={"choch_bullish": True, "choch_bearish": False},
            fvgs={}, obs={}, direction="SHORT", price=150.0,
        )
        assert short_score == 0

    def test_max_score_within_bounds(self):
        """Even perfect L1 inputs should not exceed 25 (the L1 max)."""
        perfect = _l1_smc(
            structure={"choch_bullish": True},
            fvgs={"fvg_bullish": {"top": 150.5, "bottom": 149.5}},
            obs={"ob_bullish": {"top": 150.5, "bottom": 149.5}},
            direction="LONG",
            price=150.0,
        )
        assert perfect <= 25.0


# ──────────────────────────────────────────────────────────────
# score() function — with mocked external calls
# ──────────────────────────────────────────────────────────────

class TestScoreFunction:

    def _make_analysis(self, direction="LONG", has_structure=True):
        import pandas as pd
        df = pd.DataFrame({
            "open":   [150.0] * 30,
            "high":   [152.0] * 30,
            "low":    [148.0] * 30,
            "close":  [151.0] * 30,
            "volume": [1_000_000] * 30,
        })
        return {
            "ticker":        "AAPL",
            "direction":     direction,
            "current_price": 151.0,
            "entry":         151.0,
            "stop_loss":     148.0,
            "target_one":    154.0,
            "target_two":    157.0,
            "candles":       df,
            "structure":     {"choch_bullish": True} if has_structure else {},
            "fvgs":          {},
            "obs":           {},
            "liquidity_sweep": {},
        }

    def _score(self, analysis=None, strategy="day_trade"):
        from engine.scorer import score
        if analysis is None:
            analysis = self._make_analysis()
        # Ensure analysis carries strategy_type so _l3_flow_sentiment works
        analysis = {**analysis, "strategy_type": strategy}
        with patch("engine.scorer.yf.Ticker") as mock_yf, \
             patch("engine.scorer._l3_sentiment", return_value=10.0), \
             patch("engine.scorer._l3_flow_sentiment", return_value=10.0), \
             patch("engine.scorer._l5_multiframe", return_value=8.0), \
             patch("engine.adaptive_weights.get_weights",
                   return_value={"smc": 25, "technical": 35,
                                 "sentiment": 25, "risk": 15}):
            mock_yf.return_value.news = []
            mock_yf.return_value.fast_info.last_price = 151.0
            return score(analysis, strategy_type=strategy)

    def test_score_has_total(self):
        result = self._score()
        assert "total" in result

    def test_score_has_passes_flag(self):
        result = self._score()
        assert "passes" in result
        assert isinstance(result["passes"], bool)

    def test_score_total_in_range(self):
        result = self._score()
        assert 0 <= result["total"] <= 100

    def test_score_has_breakdown(self):
        result = self._score()
        assert "breakdown" in result
        assert isinstance(result["breakdown"], dict)

    def test_score_has_direction(self):
        result = self._score()
        assert result.get("direction") in ("LONG", "SHORT")

    def test_low_l1_blocks_signal(self):
        """No SMC structure → L1 < minimum → passes=False regardless of other layers."""
        analysis = self._make_analysis(has_structure=False)
        result = self._score(analysis=analysis, strategy="day_trade")
        # Options_flow doesn't require L1; day_trade does
        assert result["passes"] is False

    def test_flow_strategy_does_not_require_l1(self):
        """options_flow skips the L1 hard gate."""
        analysis = self._make_analysis(has_structure=False)
        result = self._score(analysis=analysis, strategy="options_flow")
        # Should not immediately return passes=False for missing L1
        assert "total" in result   # just checking it doesn't crash and returns a score


# ──────────────────────────────────────────────────────────────
# Fire threshold sanity checks
# ──────────────────────────────────────────────────────────────

class TestFireThresholds:

    def test_all_thresholds_above_50(self):
        """No strategy should fire on noise (score < 50)."""
        for strategy, threshold in STRATEGY_THRESHOLDS.items():
            assert threshold > 50, f"{strategy} threshold {threshold} is too permissive"

    def test_options_flow_has_highest_or_equal_threshold(self):
        """High-risk flow strategies should require higher confidence."""
        flow_th = STRATEGY_THRESHOLDS["options_flow"]
        day_th  = STRATEGY_THRESHOLDS["day_trade"]
        assert flow_th >= day_th, "options_flow should require >= confidence vs day_trade"
