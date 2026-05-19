"""
Unit tests — engine/session_classifier.py

Covers:
  - Session mode detection for every time window
  - FOMC date blocking (1:30-2:30 PM window)
  - OpEx day detection (_is_opex_day)
  - Swing blocked on OpEx (_allows_swing)
  - Score thresholds per session
  - 9:30-9:45 is BLOCKED without catalyst, CATALYST_ONLY with catalyst
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch
from datetime import datetime, date
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


def _dt(hour, minute, *, day=14, month=5, year=2026):
    """ET datetime. May 14 2026 is a Thursday — standard trading day."""
    return datetime(year, month, day, hour, minute, tzinfo=ET)


def _classify(dt_et, has_catalyst=False):
    from engine.session_classifier import classify
    # Patch _et_now so time-based checks use our test datetime
    # Also patch date.today() so FOMC check doesn't trip on the real date
    with patch("engine.session_classifier._et_now", return_value=dt_et), \
         patch("engine.session_classifier.date") as mock_date:
        # Return a safe non-FOMC date for the "standard" tests
        mock_date.today.return_value = date(2026, 5, 14)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        return classify(has_premarket_catalyst=has_catalyst)


# ──────────────────────────────────────────────────────────────
# Session mode detection
# ──────────────────────────────────────────────────────────────

class TestSessionModes:

    def test_pre_market(self):
        result = _classify(_dt(8, 0))
        assert result["mode"] == "PRE_MARKET"

    def test_open_without_catalyst_is_blocked(self):
        """9:30-9:45 without pre-market catalyst → BLOCKED (too risky to fire)."""
        result = _classify(_dt(9, 35), has_catalyst=False)
        assert result["mode"] == "BLOCKED"

    def test_open_with_catalyst_is_catalyst_only(self):
        """9:30-9:45 WITH pre-market catalyst → CATALYST_ONLY."""
        result = _classify(_dt(9, 35), has_catalyst=True)
        assert result["mode"] == "CATALYST_ONLY"

    def test_orb_start(self):
        """9:45 ET = ORB begins."""
        result = _classify(_dt(9, 45))
        assert result["mode"] == "ORB"

    def test_orb_mid(self):
        result = _classify(_dt(9, 52))
        assert result["mode"] == "ORB"

    def test_standard_start(self):
        """10:00 ET = STANDARD session starts."""
        result = _classify(_dt(10, 0))
        assert result["mode"] == "STANDARD"

    def test_standard_mid_day(self):
        result = _classify(_dt(13, 0))
        assert result["mode"] == "STANDARD"

    def test_close_only_start(self):
        """15:30 ET = CLOSE_ONLY starts."""
        result = _classify(_dt(15, 30))
        assert result["mode"] == "CLOSE_ONLY"

    def test_close_only_mid(self):
        result = _classify(_dt(15, 45))
        assert result["mode"] == "CLOSE_ONLY"

    def test_after_hours(self):
        """16:00+ = AFTER_HOURS."""
        result = _classify(_dt(16, 0))
        assert result["mode"] == "AFTER_HOURS"

    def test_after_hours_evening(self):
        result = _classify(_dt(18, 30))
        assert result["mode"] == "AFTER_HOURS"


# ──────────────────────────────────────────────────────────────
# Session flags
# ──────────────────────────────────────────────────────────────

class TestSessionFlags:

    def test_blocked_flag_for_pre_market(self):
        result = _classify(_dt(8, 0))
        assert result["blocked"] is True

    def test_blocked_flag_for_after_hours(self):
        result = _classify(_dt(16, 30))
        assert result["blocked"] is True

    def test_not_blocked_in_standard(self):
        result = _classify(_dt(11, 0))
        assert result["blocked"] is False

    def test_market_open_true_during_session(self):
        result = _classify(_dt(11, 0))
        assert result["market_open"] is True

    def test_market_open_false_after_hours(self):
        result = _classify(_dt(17, 0))
        assert result["market_open"] is False

    def test_allows_swing_in_standard(self):
        result = _classify(_dt(11, 0))
        assert result["allows_swing"] is True

    def test_allows_swing_false_on_opex(self):
        """May 15, 2026 = 3rd Friday (OpEx) — swing blocked."""
        opex_dt = datetime(2026, 5, 15, 11, 0, tzinfo=ET)
        with patch("engine.session_classifier._et_now", return_value=opex_dt), \
             patch("engine.session_classifier.date") as mock_date:
            mock_date.today.return_value = date(2026, 5, 15)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            from engine.session_classifier import classify
            result = classify()
        assert result["allows_swing"] is False


# ──────────────────────────────────────────────────────────────
# Score thresholds per session
# ──────────────────────────────────────────────────────────────

class TestSessionThresholds:

    def _min_score(self, mode: str) -> int:
        from engine.session_classifier import SESSION_THRESHOLDS
        return SESSION_THRESHOLDS[mode]

    def test_catalyst_only_threshold_highest(self):
        assert self._min_score("CATALYST_ONLY") == 85

    def test_orb_threshold(self):
        assert self._min_score("ORB") == 80

    def test_standard_threshold(self):
        assert self._min_score("STANDARD") == 70

    def test_close_only_threshold(self):
        assert self._min_score("CLOSE_ONLY") == 80

    def test_pre_market_blocked(self):
        assert self._min_score("PRE_MARKET") >= 999

    def test_after_hours_blocked(self):
        assert self._min_score("AFTER_HOURS") >= 999

    def test_blocked_session_blocked(self):
        assert self._min_score("BLOCKED") >= 999

    def test_threshold_returned_in_classify(self):
        result = _classify(_dt(11, 0))
        assert "threshold" in result
        assert result["threshold"] == 70  # STANDARD


# ──────────────────────────────────────────────────────────────
# FOMC date blocking
# ──────────────────────────────────────────────────────────────

class TestFOMCBlocking:

    def test_fomc_date_during_window_is_blocked(self):
        """2026-04-29 is a known FOMC date. 1:45 PM ET = inside window."""
        from engine.session_classifier import classify, FOMC_DATES
        assert "2026-04-29" in FOMC_DATES, "Test requires 2026-04-29 in FOMC_DATES"

        dt = datetime(2026, 4, 29, 13, 45, tzinfo=ET)
        with patch("engine.session_classifier._et_now", return_value=dt), \
             patch("engine.session_classifier.date") as mock_date:
            mock_date.today.return_value = date(2026, 4, 29)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            result = classify()
        assert result["mode"] == "BLOCKED"

    def test_fomc_date_outside_window_not_blocked(self):
        """Same FOMC day but 10 AM ET — before announcement."""
        from engine.session_classifier import classify
        dt = datetime(2026, 4, 29, 10, 0, tzinfo=ET)
        with patch("engine.session_classifier._et_now", return_value=dt), \
             patch("engine.session_classifier.date") as mock_date:
            mock_date.today.return_value = date(2026, 4, 29)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            result = classify()
        assert result["mode"] != "BLOCKED"

    def test_fomc_day_flag_set(self):
        """classify() should set is_fomc_day=True on an FOMC date."""
        from engine.session_classifier import classify
        dt = datetime(2026, 4, 29, 10, 0, tzinfo=ET)
        with patch("engine.session_classifier._et_now", return_value=dt), \
             patch("engine.session_classifier.date") as mock_date:
            mock_date.today.return_value = date(2026, 4, 29)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            result = classify()
        assert result["is_fomc_day"] is True


# ──────────────────────────────────────────────────────────────
# OpEx detection
# ──────────────────────────────────────────────────────────────

class TestOpEx:

    def test_third_friday_is_opex(self):
        from engine.session_classifier import _is_opex_day
        dt = datetime(2026, 5, 15, 10, 0, tzinfo=ET)  # 3rd Friday May 2026
        assert _is_opex_day(dt) is True

    def test_first_friday_not_opex(self):
        from engine.session_classifier import _is_opex_day
        dt = datetime(2026, 5, 1, 10, 0, tzinfo=ET)  # 1st Friday
        assert _is_opex_day(dt) is False

    def test_fourth_friday_not_opex(self):
        from engine.session_classifier import _is_opex_day
        dt = datetime(2026, 5, 22, 10, 0, tzinfo=ET)  # 4th Friday
        assert _is_opex_day(dt) is False

    def test_wednesday_not_opex(self):
        from engine.session_classifier import _is_opex_day
        dt = datetime(2026, 5, 14, 10, 0, tzinfo=ET)  # Thursday
        assert _is_opex_day(dt) is False

    def test_opex_flag_in_classify_on_opex_day(self):
        """classify() should set is_opex_day=True on the 3rd Friday."""
        from engine.session_classifier import classify
        dt = datetime(2026, 5, 15, 10, 0, tzinfo=ET)
        with patch("engine.session_classifier._et_now", return_value=dt), \
             patch("engine.session_classifier.date") as mock_date:
            mock_date.today.return_value = date(2026, 5, 15)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            result = classify()
        assert result["is_opex_day"] is True
