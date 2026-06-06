"""Unit tests — Phase 2 threat radar pure scoring + flags. Additive; no existing
behavior touched."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine.phase2 import threat_radar as tr
from engine.phase2 import flags


def test_flags_defaults_invasive_off_readonly_on():
    # read-only safe module on; broker/invasive off
    assert flags.enabled("threat_radar") is True
    assert flags.enabled("portfolio_doctor") is False
    assert flags.enabled("unknown_thing") is False


def test_flags_env_override(monkeypatch):
    monkeypatch.setenv("PHASE2_PORTFOLIO_DOCTOR", "true")
    assert flags.enabled("portfolio_doctor") is True
    monkeypatch.setenv("PHASE2_THREAT_RADAR", "off")
    assert flags.enabled("threat_radar") is False


def test_vix_threat_scaling():
    assert tr._vix_threat(12, 0) == 0          # calm
    assert tr._vix_threat(22, 0) == 45
    assert tr._vix_threat(35, 0) == 90         # panic
    assert tr._vix_threat(22, 15) == 65        # +20 for intraday spike
    assert tr._vix_threat(None, None) == 40    # fail-safe neutral


def test_trend_and_breadth_threat():
    assert tr._trend_threat(True, -2) == 10    # uptrend near highs
    assert tr._trend_threat(False, -25) == 90  # deep below 200dma
    assert tr._breadth_threat(80) == 0         # healthy breadth
    assert tr._breadth_threat(20) == 80        # narrow
    assert tr._breadth_threat(None) == 40


def test_aggregate_green_vs_red():
    calm = [
        {"key": "vix", "label": "VIX", "weight": 0.5, "threat": 5, "detail": "12"},
        {"key": "trend", "label": "Trend", "weight": 0.5, "threat": 10, "detail": "up"},
    ]
    g = tr._aggregate(calm)
    assert g["level"] == "GREEN" and g["threat_score"] < 25 and "calm" in g["summary"].lower()

    risky = [
        {"key": "vix", "label": "Volatility (VIX)", "weight": 0.5, "threat": 90, "detail": "VIX 32"},
        {"key": "breadth", "label": "Breadth", "weight": 0.5, "threat": 80, "detail": "20% above 50-DMA"},
    ]
    r = tr._aggregate(risky)
    assert r["level"] == "RED" and r["threat_score"] >= 75
    assert any("Volatility" in x for x in r["reasons"])     # top risk surfaced


def test_compute_never_raises(monkeypatch):
    # even with every data source broken, compute returns a structured payload
    import engine.regime_detector as rd
    monkeypatch.setattr(rd, "detect", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    out = tr.compute()
    assert "level" in out and out.get("enabled") is True
