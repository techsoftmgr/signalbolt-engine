"""
Unit tests — engine.signal_telemetry.capture (logging-only fire-time telemetry).

Guarantees: it records regime context + sector + concentration counts, the
counts are taken from `signals` filtered by status/direction(/strategy), and it
ALWAYS fails open (never raises, returns ("", {}) worst case) so it can never
block a signal fire.
"""
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import signal_telemetry as st


def _fake_sb(count_value=3):
    """A Supabase stub whose .select().eq()...limit().execute().count == count_value."""
    sb = MagicMock()
    chain = MagicMock()
    chain.eq.return_value = chain
    chain.limit.return_value = chain
    exec_res = MagicMock()
    exec_res.count = count_value
    chain.execute.return_value = exec_res
    sb.table.return_value.select.return_value = chain
    return sb


def setup_function(_):
    # Clear the module regime cache between tests.
    st._regime_cache["data"] = None
    st._regime_cache["ts"] = 0.0


def test_capture_records_regime_sector_and_concentration():
    fake_regime = {"regime_type": "TRENDING_BULL", "vix": 14.2, "vix_change_pct": -0.03,
                   "adx": 27.0, "above_200ma": True}
    with patch("engine.regime_detector.detect", return_value=fake_regime), \
         patch("engine.risk_manager.get_sector", return_value="Technology"):
        regime_type, study = st.capture(_fake_sb(5), "NVDA", "SHORT", "breakdown")

    assert regime_type == "TRENDING_BULL"
    assert study["regime_type"] == "TRENDING_BULL"
    assert study["vix"] == 14.2
    assert study["adx"] == 27.0
    assert study["spy_above_200ma"] is True
    assert study["sector"] == "Technology"
    # concentration counts present (taken before this signal is inserted)
    assert study["open_dir_total"] == 5
    assert study["open_strat"] == 5


def test_capture_fails_open_when_regime_detect_raises():
    with patch("engine.regime_detector.detect", side_effect=RuntimeError("alpaca down")), \
         patch("engine.risk_manager.get_sector", return_value="Energy"):
        regime_type, study = st.capture(_fake_sb(0), "XOM", "SHORT", "breakdown")
    # Regime missing → empty string, but the call still returns and other fields fill.
    assert regime_type == ""
    assert study.get("sector") == "Energy"
    assert study["open_dir_total"] == 0


def test_count_returns_none_when_sb_none():
    assert st._count(None, status="active") is None


def test_capture_handles_sb_none_without_counts():
    fake_regime = {"regime_type": "RANGING", "vix": 18.0, "above_200ma": False}
    with patch("engine.regime_detector.detect", return_value=fake_regime), \
         patch("engine.risk_manager.get_sector", return_value="Other"):
        regime_type, study = st.capture(None, "ZZZZ", "SHORT", "breakdown")
    assert regime_type == "RANGING"
    assert study["open_dir_total"] is None      # no sb → count fails open to None
    assert study["open_strat"] is None


def test_regime_is_cached_within_ttl():
    fake_regime = {"regime_type": "LOW_VOL", "vix": 12.0}
    with patch("engine.regime_detector.detect", return_value=fake_regime) as det, \
         patch("engine.risk_manager.get_sector", return_value="Other"):
        st.capture(_fake_sb(), "AAA", "SHORT", "breakdown")
        st.capture(_fake_sb(), "BBB", "SHORT", "breakdown")
    # Two captures, one detect() call (cached within the 180s TTL).
    assert det.call_count == 1
