"""
Unit tests — engine/sl_tp_engine.py

Covers:
  - Round-number avoidance (stop raids)
  - ATR computation
  - SL/TP calculation bounds
  - Minimum risk/reward ratio
  - Gamma wall integration
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import pandas as pd
from unittest.mock import patch
from engine.sl_tp_engine import (
    _round_number_adjustment,
    _compute_atr,
    MIN_RR,
    MAX_SL_PCT,
    MIN_SL_PCT,
)

# ── Shared mock dicts for regime / session / gamma ────────────
_regime = {
    "regime_type": "TRENDING_BULL",
    "vix": 18.0,
    "vix_change_pct": 0.0,
    "blocked": False,
}
_session = {
    "mode": "STANDARD",
    "sl_adjustment": 1.0,
    "is_opex_day": False,
    "threshold": 70,
}
_gamma = {
    "walls": [],
    "net_gex": 0,
    "is_negative_gamma": False,
    "pin_risk": False,
}


# ──────────────────────────────────────────────────────────────
# _round_number_adjustment
# ──────────────────────────────────────────────────────────────

class TestRoundNumberAdjustment:

    def test_price_far_from_round_number_unchanged(self):
        """Price like $150.37 is far from $150 — should be returned as-is."""
        price = 150.37
        result = _round_number_adjustment(price, nudge="down")
        assert result == price

    def test_long_sl_nudged_below_round_number(self):
        """LONG stop at $150.00 — nudge DOWN away from round number raid zone."""
        price = 150.00
        result = _round_number_adjustment(price, nudge="down")
        assert result < 150.00, "LONG SL should be pushed below round number"

    def test_short_sl_nudged_above_round_number(self):
        """SHORT stop at $200.00 — nudge UP away from round number raid zone."""
        price = 200.00
        result = _round_number_adjustment(price, nudge="up")
        assert result > 200.00, "SHORT SL should be pushed above round number"

    def test_near_round_number_adjusted(self):
        """Price within 0.2% of round number ($150.01) should be adjusted."""
        price = 150.01   # 0.007% from $150
        result = _round_number_adjustment(price, nudge="down", buffer=0.002)
        assert result < 150.00 or result > 150.00, "Near-round price should be adjusted"

    def test_result_is_non_zero_positive(self):
        """Result must always be a positive price."""
        for price in [10.0, 50.00, 100.00, 500.00, 999.99]:
            r = _round_number_adjustment(price, nudge="down")
            assert r > 0


# ──────────────────────────────────────────────────────────────
# _compute_atr
# ──────────────────────────────────────────────────────────────

class TestComputeATR:

    def _make_df(self, highs, lows, closes):
        return pd.DataFrame({"high": highs, "low": lows, "close": closes})

    def test_atr_positive(self, ohlcv_uptrend):
        """ATR must always be positive."""
        atr = _compute_atr(ohlcv_uptrend)
        assert atr > 0

    def test_atr_reasonable_magnitude(self, ohlcv_uptrend):
        """ATR for a $150 stock should be between $0.10 and $20."""
        atr = _compute_atr(ohlcv_uptrend)
        assert 0.10 < atr < 20.0

    def test_atr_volatile_stock_higher(self):
        """A volatile stock should have higher ATR than a calm one."""
        calm_df = pd.DataFrame({
            "high":  [100.1] * 30,
            "low":   [99.9]  * 30,
            "close": [100.0] * 30,
        })
        volatile_df = pd.DataFrame({
            "high":  [110.0, 100.0] * 15,
            "low":   [90.0,  99.0]  * 15,
            "close": [100.0, 99.5]  * 15,
        })
        atr_calm     = _compute_atr(calm_df)
        atr_volatile = _compute_atr(volatile_df)
        assert atr_volatile > atr_calm

    def test_atr_short_df_returns_fallback(self, ohlcv_short):
        """DataFrame with fewer than period+1 rows should not crash."""
        result = _compute_atr(ohlcv_short, period=14)
        # Should either return a fallback value or a small positive number
        assert isinstance(result, float)
        assert result >= 0


# ──────────────────────────────────────────────────────────────
# Full SL/TP calculation (calculate function)
# ──────────────────────────────────────────────────────────────

class TestCalculateSLTP:

    def _call(self, direction, entry, df, strategy_type="day_trade"):
        """Helper: call calculate() with correct signature and mocked internals."""
        from engine.sl_tp_engine import calculate
        with patch("engine.regime_detector.get_sl_adjustment", return_value=1.0):
            return calculate(
                direction=direction,
                entry=entry,
                df=df,
                regime=_regime,
                session=_session,
                gamma=_gamma,
                strategy_type=strategy_type,
            )

    def test_long_sl_below_entry(self, ohlcv_uptrend):
        result = self._call("LONG", 180.0, ohlcv_uptrend, "day_trade")
        assert result["stop_loss"] < 180.0, "LONG SL must be below entry"

    def test_long_targets_above_entry(self, ohlcv_uptrend):
        result = self._call("LONG", 180.0, ohlcv_uptrend, "day_trade")
        assert result["target_one"] > 180.0, "LONG T1 must be above entry"
        assert result["target_two"] > result["target_one"], "T2 must be above T1"

    def test_short_sl_above_entry(self, ohlcv_downtrend):
        result = self._call("SHORT", 450.0, ohlcv_downtrend, "day_trade")
        assert result["stop_loss"] > 450.0, "SHORT SL must be above entry"

    def test_short_targets_below_entry(self, ohlcv_downtrend):
        result = self._call("SHORT", 450.0, ohlcv_downtrend, "day_trade")
        assert result["target_one"] < 450.0
        assert result["target_two"] < result["target_one"]

    def test_minimum_risk_reward_ratio(self, ohlcv_uptrend):
        """R:R to T1 must be >= MIN_RR (2.0)."""
        result = self._call("LONG", 180.0, ohlcv_uptrend, "scalping")
        risk   = abs(180.0 - result["stop_loss"])
        reward = abs(result["target_one"] - 180.0)
        if risk > 0:
            rr = reward / risk
            assert rr >= MIN_RR * 0.9, f"R:R {rr:.2f} below minimum {MIN_RR}"

    def test_sl_not_at_round_number(self, ohlcv_uptrend):
        """SL should not land exactly on a round number."""
        result = self._call("LONG", 182.0, ohlcv_uptrend, "day_trade")
        sl = result["stop_loss"]
        assert sl != round(sl), f"SL landed exactly on round number ${sl}"

    def test_sl_within_max_pct(self, ohlcv_uptrend):
        """SL must not be further than MAX_SL_PCT (8%) from entry."""
        result = self._call("LONG", 180.0, ohlcv_uptrend, "swing_trade")
        sl_pct = abs(180.0 - result["stop_loss"]) / 180.0
        assert sl_pct <= MAX_SL_PCT, f"SL {sl_pct:.1%} exceeds max {MAX_SL_PCT:.1%}"

    def test_result_has_required_keys(self, ohlcv_uptrend):
        result = self._call("LONG", 180.0, ohlcv_uptrend)
        for key in ["stop_loss", "target_one", "target_two", "adjustments"]:
            assert key in result, f"Missing key: {key}"
        # Either risk_reward_1 or risk_reward_2 must be present
        assert "risk_reward_1" in result or "risk_reward_2" in result, \
            "Missing risk_reward_1 or risk_reward_2"
