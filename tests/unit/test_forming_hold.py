"""
Unit tests — forming detectors inherit the parent swing hold-window.

Predictive *_forming signals were falling back to the 48h default (expired in
~2 days). They now share the 240h / 10-day swing window with their confirmed
parents so a developing pattern isn't time-expired early.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime, timezone, timedelta
import pytest

from engine import runner

_FORMING = ["breakdown_forming", "distrib_forming", "peak_forming",
            "turn_forming", "accum_forming"]


class TestFormingMaxHold:
    def test_all_forming_are_swing_window(self):
        for s in _FORMING:
            assert runner.STRATEGY_MAX_HOLD_HOURS.get(s) == 240.0, s

    def test_forming_matches_its_parent(self):
        """Same created time → forming and confirmed parent expire identically."""
        created = datetime.now(timezone.utc) - timedelta(days=3)
        pairs = [("peak_forming", "peak"), ("turn_forming", "turnaround"),
                 ("breakdown_forming", "breakdown")]
        for child, parent in pairs:
            assert runner.is_past_max_hold(created, child) == \
                   runner.is_past_max_hold(created, parent), f"{child} vs {parent}"

    def test_three_day_old_forming_not_expired(self):
        """3 days in is well within the 10-trading-day window (was expired at 2d)."""
        created = datetime.now(timezone.utc) - timedelta(days=3)
        assert runner.is_past_max_hold(created, "peak_forming") is False

    def test_unknown_strategy_still_defaults_48h(self):
        """The 48h fallback is unchanged for genuinely unknown strategies."""
        assert runner.STRATEGY_MAX_HOLD_HOURS.get("totally_unknown") is None
