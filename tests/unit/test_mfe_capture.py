"""
Unit tests — signal_monitor._capture_excursion records running MFE (peak
favorable) / MAE (peak adverse) unrealized % on a signal, measurement-only,
writing the row only when a new extreme is set. Powers profit give-back
(peak − realized) measurement, incl. after-hours / pre-market excursions.
"""
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import signal_monitor


def _fake_sb():
    sb = MagicMock()
    # sb.table("signals").update({...}).eq("id", x).execute()
    sb.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()
    return sb


def _captured(sb):
    """Return the dict passed to the last .update(...) call, or None."""
    upd = sb.table.return_value.update
    return upd.call_args[0][0] if upd.called else None


def test_long_sets_first_extreme():
    sb = _fake_sb()
    sig = {"id": "1", "ticker": "AAA", "direction": "LONG", "entry_price": 100.0,
           "score_breakdown": {"detector_source": "BREAKOUT"}}
    with patch.object(signal_monitor, "_current_price", return_value=103.0):
        signal_monitor._capture_excursion(sb, sig)
    cap = _captured(sb)["score_breakdown"]
    assert cap["mfe_pct"] == 3.0 and cap["mae_pct"] == 3.0
    # local copy refreshed
    assert sig["score_breakdown"]["mfe_pct"] == 3.0


def test_short_pnl_direction():
    sb = _fake_sb()
    sig = {"id": "2", "ticker": "BBB", "direction": "SHORT", "entry_price": 100.0,
           "score_breakdown": {}}
    with patch.object(signal_monitor, "_current_price", return_value=96.0):  # short +4%
        signal_monitor._capture_excursion(sb, sig)
    cap = _captured(sb)["score_breakdown"]
    assert cap["mfe_pct"] == 4.0 and cap["mae_pct"] == 4.0


def test_updates_mae_keeps_mfe_high():
    sb = _fake_sb()
    sig = {"id": "3", "ticker": "CCC", "direction": "LONG", "entry_price": 100.0,
           "score_breakdown": {"mfe_pct": 5.0, "mae_pct": 1.0}}
    with patch.object(signal_monitor, "_current_price", return_value=98.0):  # -2% now
        signal_monitor._capture_excursion(sb, sig)
    cap = _captured(sb)["score_breakdown"]
    assert cap["mfe_pct"] == 5.0      # peak preserved
    assert cap["mae_pct"] == -2.0     # new worst


def test_no_new_extreme_skips_write():
    sb = _fake_sb()
    sig = {"id": "4", "ticker": "DDD", "direction": "LONG", "entry_price": 100.0,
           "score_breakdown": {"mfe_pct": 5.0, "mae_pct": -3.0}}
    with patch.object(signal_monitor, "_current_price", return_value=102.0):  # +2%, within band
        signal_monitor._capture_excursion(sb, sig)
    assert not sb.table.return_value.update.called   # no new extreme → no DB write


def test_phantom_print_rejected():
    sb = _fake_sb()
    sig = {"id": "5", "ticker": "EEE", "direction": "LONG", "entry_price": 100.0,
           "score_breakdown": {}}
    with patch.object(signal_monitor, "_current_price", return_value=500.0):  # +400% bad print
        signal_monitor._capture_excursion(sb, sig)
    assert not sb.table.return_value.update.called   # implausible → skipped


def test_no_price_is_noop():
    sb = _fake_sb()
    sig = {"id": "6", "ticker": "FFF", "direction": "LONG", "entry_price": 100.0, "score_breakdown": {}}
    with patch.object(signal_monitor, "_current_price", return_value=None):
        signal_monitor._capture_excursion(sb, sig)
    assert not sb.table.return_value.update.called
