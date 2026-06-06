"""Unit test — signal_telemetry.live_regime_type never returns empty (so detector
signals are always regime-sliceable), prefers detect, then the worker cache,
then a neutral default."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine import signal_telemetry as st


def test_live_regime_prefers_detected(monkeypatch):
    monkeypatch.setattr(st, "get_regime", lambda: {"regime_type": "PANIC", "vix": 30})
    assert st.live_regime_type() == "PANIC"


def test_live_regime_falls_back_to_worker_cache(monkeypatch):
    import engine.stream as stream_mod
    monkeypatch.setattr(st, "get_regime", lambda: {})                       # detect empty
    monkeypatch.setattr(stream_mod, "_get_regime", lambda: {"regime_type": "HIGH_VOL"})
    assert st.live_regime_type() == "HIGH_VOL"


def test_live_regime_default_when_all_empty(monkeypatch):
    import engine.stream as stream_mod
    monkeypatch.setattr(st, "get_regime", lambda: {})
    monkeypatch.setattr(stream_mod, "_get_regime", lambda: {})
    assert st.live_regime_type(default="RANGING") == "RANGING"
    assert st.live_regime_type() != ""                                      # never empty


def test_live_regime_never_raises(monkeypatch):
    import engine.stream as stream_mod
    def boom(): raise RuntimeError("x")
    monkeypatch.setattr(st, "get_regime", boom)
    monkeypatch.setattr(stream_mod, "_get_regime", lambda: {})
    assert st.live_regime_type(default="NEUTRALISH") == "NEUTRALISH"
