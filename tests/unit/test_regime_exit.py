"""Unit tests — regime_exit decision brain (Layer 3). Pure; not wired live."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine import regime_exit as rx


def test_not_wired_by_default():
    assert rx.ENFORCE is False              # ready but NOT used


def test_long_adverse_flip_in_profit_tightens():
    r = rx.assess("LONG", "TRENDING_BULL", "PANIC", 5.0)   # RISK_ON → RISK_OFF, +5%
    assert r["action"] == "TIGHTEN" and r["flip_from"] == "RISK_ON" and r["flip_to"] == "RISK_OFF"


def test_long_adverse_flip_in_red_exits():
    r = rx.assess("LONG", "LOW_VOL", "TRENDING_BEAR", -2.0)  # RISK_ON → RISK_OFF, red
    assert r["action"] == "EXIT"


def test_long_no_flip_holds():
    assert rx.assess("LONG", "TRENDING_BULL", "LOW_VOL", 3.0)["action"] == "HOLD"   # both RISK_ON


def test_long_favorable_flip_holds():
    # regime got MORE risk-on for a long → not adverse
    assert rx.assess("LONG", "RANGING", "TRENDING_BULL", 1.0)["action"] == "HOLD"


def test_short_adverse_is_risk_on_move():
    r = rx.assess("SHORT", "PANIC", "TRENDING_BULL", 4.0)   # RISK_OFF → RISK_ON, +4% (adverse for short)
    assert r["action"] == "TIGHTEN"
    # a short in a deepening risk-off is NOT adverse → HOLD
    assert rx.assess("SHORT", "HIGH_VOL", "PANIC", 2.0)["action"] == "HOLD"


def test_unknown_regime_holds():
    assert rx.assess("LONG", "", "PANIC", 1.0)["action"] == "HOLD"
    assert rx.assess("LONG", "TRENDING_BULL", None, 1.0)["action"] == "HOLD"
    assert rx.adverse_flip("LONG", "ANY", "RISK_OFF") is False
