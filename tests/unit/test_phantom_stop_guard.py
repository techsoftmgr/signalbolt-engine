"""
Unit tests — alpaca_client.confirm_level_cross (phantom-stop guard).

Regression cover for the 2026-06-03 incident: a single bad SIP last-trade print
booked fake stop-outs (CMCSA "stop @ 26.50" while the 1-min high was 23.72; BA
"stop @ 230" while its whole-day high was 217.72). A level cross must be
corroborated by recent 1-min bars OR a fresh 2nd read before a close is booked.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch
import pandas as pd
import pytest

from engine import alpaca_client as ac


def _bars(high, low):
    """Minimal 1-min bar frame with the given high/low extremes."""
    return pd.DataFrame({
        "open":   [low, high],
        "high":   [high * 0.99, high],
        "low":    [low, low * 1.01],
        "close":  [(high + low) / 2, (high + low) / 2],
        "volume": [1000, 1000],
    })


class TestConfirmLevelCross:
    # ── Phantom incident: SHORT stop, tape never reached the stop ──────────────
    def test_short_stop_phantom_rejected(self):
        """CMCSA: stop 24.62, real 1-min high 23.715, 2nd-read 23.62 → NOT confirmed."""
        with patch.object(ac, "get_bars", return_value=_bars(23.715, 23.58)), \
             patch.object(ac, "get_latest_price", return_value=23.62):
            assert ac.confirm_level_cross("CMCSA", 24.62, is_long=False, kind="stop") is False

    def test_short_stop_phantom_rejected_no_bars(self):
        """BA-style: bars unavailable, 2nd-read (213.56) far below the 225.5 stop → reject."""
        with patch.object(ac, "get_bars", return_value=None), \
             patch.object(ac, "get_latest_price", return_value=213.56):
            assert ac.confirm_level_cross("BA", 225.5, is_long=False, kind="stop") is False

    # ── Real breaches are confirmed ────────────────────────────────────────────
    def test_short_stop_confirmed_by_bars(self):
        """Bar high actually reached the stop → confirmed."""
        with patch.object(ac, "get_bars", return_value=_bars(24.80, 23.50)), \
             patch.object(ac, "get_latest_price", return_value=23.60):
            assert ac.confirm_level_cross("CMCSA", 24.62, is_long=False, kind="stop") is True

    def test_short_stop_confirmed_by_second_read(self):
        """Bars unavailable but a fresh read confirms a sustained move → confirmed."""
        with patch.object(ac, "get_bars", return_value=None), \
             patch.object(ac, "get_latest_price", return_value=24.90):
            assert ac.confirm_level_cross("CMCSA", 24.62, is_long=False, kind="stop") is True

    def test_long_stop_confirmed_by_bars(self):
        """LONG stop: bar low fell to/below the stop → confirmed."""
        with patch.object(ac, "get_bars", return_value=_bars(101.0, 98.5)), \
             patch.object(ac, "get_latest_price", return_value=99.5):
            assert ac.confirm_level_cross("AAPL", 99.0, is_long=True, kind="stop") is True

    def test_long_target_confirmed_by_bars(self):
        """LONG target: bar high reached the target → confirmed."""
        with patch.object(ac, "get_bars", return_value=_bars(105.0, 100.0)), \
             patch.object(ac, "get_latest_price", return_value=104.5):
            assert ac.confirm_level_cross("AAPL", 104.0, is_long=True, kind="target") is True

    def test_short_target_confirmed_by_bars(self):
        """SHORT target: bar low reached the target → confirmed."""
        with patch.object(ac, "get_bars", return_value=_bars(80.0, 78.0)), \
             patch.object(ac, "get_latest_price", return_value=79.0):
            assert ac.confirm_level_cross("NFLX", 78.5, is_long=False, kind="target") is True

    # ── Fail-closed when no data at all ────────────────────────────────────────
    def test_no_data_fails_closed(self):
        """No bars and no last-trade → do NOT confirm (never fabricate a close)."""
        with patch.object(ac, "get_bars", return_value=None), \
             patch.object(ac, "get_latest_price", return_value=None):
            assert ac.confirm_level_cross("CMCSA", 24.62, is_long=False, kind="stop") is False

    def test_long_stop_not_reached_rejected(self):
        """LONG stop: neither bars nor read fell to the stop → reject."""
        with patch.object(ac, "get_bars", return_value=_bars(105.0, 101.0)), \
             patch.object(ac, "get_latest_price", return_value=102.0):
            assert ac.confirm_level_cross("AAPL", 99.0, is_long=True, kind="stop") is False


class TestSaneClosePrice:
    """Bad-print guard for NON-level closes (EOD / time-stop / trend exit)."""

    def test_gross_high_outlier_clamped(self):
        """A close print far ABOVE the recent range is clamped to the high."""
        with patch.object(ac, "get_bars", return_value=_bars(23.72, 23.58)):
            # 26.50 is ~12% above the 23.72 high → clamp to 23.72
            assert ac.sane_close_price("CMCSA", 26.50) == pytest.approx(23.72)

    def test_gross_low_outlier_clamped(self):
        """A close print far BELOW the recent range is clamped to the low."""
        with patch.object(ac, "get_bars", return_value=_bars(760.0, 754.0)):
            # SPY phantom 633 is ~16% below the 754 low → clamp to 754
            assert ac.sane_close_price("SPY", 633.87) == pytest.approx(754.0)

    def test_in_range_price_unchanged(self):
        """A normal price within tolerance passes through untouched."""
        with patch.object(ac, "get_bars", return_value=_bars(24.0, 23.5)):
            assert ac.sane_close_price("CMCSA", 23.7) == pytest.approx(23.7)

    def test_small_move_within_tolerance_unchanged(self):
        """A price slightly outside the range (<5%) is a real move — kept."""
        with patch.object(ac, "get_bars", return_value=_bars(24.0, 23.5)):
            # 24.5 is ~2% above the high → within 5% tolerance → unchanged
            assert ac.sane_close_price("CMCSA", 24.5) == pytest.approx(24.5)

    def test_no_bars_returns_raw(self):
        """Fail-open: no bars → return the raw price (never block a close)."""
        with patch.object(ac, "get_bars", return_value=None):
            assert ac.sane_close_price("CMCSA", 26.50) == pytest.approx(26.50)

    def test_none_price_returns_none(self):
        assert ac.sane_close_price("CMCSA", None) is None
