"""
Unit test — swing-class strategies are EXCLUDED from intraday early-booking.

Regression for the AMZN breakdown that got booked at +1.2% the morning after
(the early-book "stalling >180 calendar min" rule fires overnight). Only
"swing_trade" was excluded before; all daily-swing detectors must be too.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import signal_monitor as sm


class TestSwingEarlyBookExclusion:
    def test_all_daily_swing_detectors_excluded(self):
        for s in ("swing_trade", "breakdown", "breakout", "turnaround", "peak",
                  "breakdown_forming", "distrib_forming", "peak_forming",
                  "turn_forming", "accum_forming", "position_trade"):
            assert s in sm._SWING_LIKE_STRATEGIES, s

    def test_intraday_strategies_still_get_early_book(self):
        for s in ("scalping", "day_trade"):
            assert s not in sm._SWING_LIKE_STRATEGIES, s
