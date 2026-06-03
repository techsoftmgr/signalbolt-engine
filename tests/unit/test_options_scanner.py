"""
Unit tests — engine/options_scanner.py strike + DTE alignment.

Options for the 1-10 day swing detectors should target ~2-4 week expiries and a
slightly IN-the-money strike (delta ~0.6) so the premium tracks the underlying
move ~1:1 with less theta — not the old 21-60 DTE / 2% OTM picks.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from engine import options_scanner as osc


class TestTargetStrike:
    def test_call_target_is_itm_below_spot(self):
        # CALL ITM = strike below spot
        assert osc._target_strike(100.0, is_call=True) == pytest.approx(98.0)

    def test_put_target_is_itm_above_spot(self):
        # PUT ITM = strike above spot
        assert osc._target_strike(100.0, is_call=False) == pytest.approx(102.0)

    def test_call_below_put_above_for_same_spot(self):
        spot = 250.0
        call = osc._target_strike(spot, True)
        put  = osc._target_strike(spot, False)
        assert call < spot < put

    def test_offset_is_two_percent(self):
        assert osc._ITM_OFFSET == pytest.approx(0.02)


class TestDteWindow:
    def test_window_is_two_to_four_weeks(self):
        assert osc._MIN_DTE == 14   # ~2 weeks
        assert osc._MAX_DTE == 30   # ~4 weeks

    def test_window_survives_max_swing_hold(self):
        # breakdown/breakout max hold is 10 trading days (~14 calendar days);
        # the option floor must cover that.
        assert osc._MIN_DTE >= 14

    def test_not_zero_dte(self):
        assert osc._MIN_DTE > 1   # never daily/0DTE for a multi-day swing
