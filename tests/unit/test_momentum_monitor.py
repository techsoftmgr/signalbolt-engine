"""
Unit tests for the profit-scaled chandelier trail in momentum_monitor.

Covers _atr_mult_for_gain (the tier ladder) and the giveback-floor math —
pure functions, no DB / network. The trail stays 100% daily-close; these tests
just pin the "tighten as the gain grows, never give back more than the cap"
behaviour so a future tweak can't silently loosen a big-winner's stop.
"""
from engine import momentum_monitor as mm


class TestAtrMultForGain:
    def test_small_gain_keeps_full_breathing_room(self):
        assert mm._atr_mult_for_gain(0.0) == 3.0
        assert mm._atr_mult_for_gain(10.0) == 3.0
        assert mm._atr_mult_for_gain(24.9) == 3.0

    def test_mid_gain_steps_in(self):
        assert mm._atr_mult_for_gain(25.0) == 2.5
        assert mm._atr_mult_for_gain(40.0) == 2.5
        assert mm._atr_mult_for_gain(49.9) == 2.5

    def test_large_gain_tightens_most(self):
        assert mm._atr_mult_for_gain(50.0) == 2.0
        assert mm._atr_mult_for_gain(120.0) == 2.0

    def test_monotonic_non_increasing(self):
        # Multiple must never widen as the gain grows.
        prev = mm._atr_mult_for_gain(-5.0)
        for g in range(0, 200, 5):
            cur = mm._atr_mult_for_gain(float(g))
            assert cur <= prev
            prev = cur


class TestGivebackFloor:
    """Replicates the LONG giveback-floor math used in manage()."""

    @staticmethod
    def _floored_chandelier(roll_high, atr, last_close, entry):
        gain = (last_close - entry) / entry * 100
        mult = mm._atr_mult_for_gain(gain)
        chand = roll_high - mult * atr
        if gain >= mm._GIVEBACK_MIN_GAIN:
            chand = max(chand, last_close * (1 - mm._GIVEBACK_CAP))
        return chand, gain

    def test_floor_never_above_last_close(self):
        # The floor can only ratchet the stop UP toward (1-cap)*close — it must
        # always stay below the close so it can't trigger a spurious exit.
        chand, gain = self._floored_chandelier(
            roll_high=300.0, atr=40.0, last_close=290.0, entry=200.0)
        assert chand < 290.0
        assert gain >= mm._GIVEBACK_MIN_GAIN

    def test_floor_binds_on_high_vol_name(self):
        # ATR so wide that 2.5×ATR sits below the 20% giveback floor → floor wins.
        # close=100, entry=70 (gain ~43% → mult 2.5), atr=15 → raw=roll_high-37.5
        chand, _ = self._floored_chandelier(
            roll_high=100.0, atr=15.0, last_close=100.0, entry=70.0)
        assert chand == 100.0 * (1 - mm._GIVEBACK_CAP)   # 80.0, floor binds

    def test_floor_dormant_below_min_gain(self):
        # Under the min gain, the floor is not applied — pure chandelier.
        chand, gain = self._floored_chandelier(
            roll_high=110.0, atr=10.0, last_close=108.0, entry=100.0)
        assert gain < mm._GIVEBACK_MIN_GAIN
        assert chand == 110.0 - 3.0 * 10.0   # full 3×ATR, no floor
