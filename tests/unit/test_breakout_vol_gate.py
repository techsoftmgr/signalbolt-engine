"""Unit tests — confirmed-breakout volume gate. Offline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import breakout_alerts as ba


def test_breakout_vol_gate(monkeypatch):
    monkeypatch.setattr(ba, "_BREAKOUT_MIN_RVOL", 1.5)
    assert ba._breakout_vol_ok({"relativeVolume": 2.0}) is True      # strong
    assert ba._breakout_vol_ok({"relativeVolume": 1.5}) is True      # exactly at threshold
    assert ba._breakout_vol_ok({"relativeVolume": 0.5}) is False     # weak — blocked
    assert ba._breakout_vol_ok({"relativeVolume": 1.2}) is False     # below threshold
    # unknown volume → fail OPEN (don't block on missing data)
    assert ba._breakout_vol_ok({"relativeVolume": None}) is True
    assert ba._breakout_vol_ok({}) is True


def test_breakout_vol_gate_disabled(monkeypatch):
    monkeypatch.setattr(ba, "_BREAKOUT_MIN_RVOL", 0.0)
    assert ba._breakout_vol_ok({"relativeVolume": 0.1}) is True      # gate off → all pass
