"""Unit tests — chart_read._scenarios builds a neutral two-sided IF/THEN plan:
reclaim level = bullish, lose level = bearish, each with a next target."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine.chart_read import _scenarios


def test_basic_bull_bear_from_levels():
    px = 100.0
    lv = {"resistance": 105.0, "support": 95.0}
    fib = {"direction": "down", "swingLow": 94.0, "swingHigh": 130.0,
           "levels": [{"price": 108.0}, {"price": 96.0}],
           "extensions": [{"price": 88.0}, {"price": 80.0}], "goldenPocket": {"low": 110, "high": 118}}
    sc = _scenarios(px, lv, fib, [])
    assert sc["bull"]["trigger"] == 105.0      # reclaim resistance
    assert sc["bear"]["trigger"] == 95.0       # lose support
    assert sc["bear"]["target"] == 88.0        # 1.272 down-extension
    assert "reclaim $105.0 = bullish" in sc["summary"]
    assert "lose $95.0 = bearish" in sc["summary"]


def test_double_bottom_neckline_is_bull_trigger():
    px = 100.0
    lv = {"resistance": 106.0, "support": 95.0}
    pats = [{"type": "Double Bottom", "tone": "bullish", "neckline": 104.0, "target": 113.0}]
    sc = _scenarios(px, lv, None, pats)
    assert sc["bull"]["trigger"] == 104.0      # neckline overrides plain resistance
    assert sc["bull"]["target"] == 113.0


def test_swing_low_fallback_when_no_support():
    px = 100.0
    lv = {"resistance": 105.0, "support": None}
    fib = {"direction": "down", "swingLow": 98.0, "swingHigh": 140.0, "levels": [],
           "extensions": [{"price": 90.0}, {"price": 84.0}], "goldenPocket": {"low": 115, "high": 122}}
    sc = _scenarios(px, lv, fib, [])
    assert sc["bear"]["trigger"] == 98.0       # falls back to the swing low


def test_none_when_no_levels():
    assert _scenarios(100.0, {"resistance": None, "support": None}, None, []) is None


def test_bear_target_below_trigger_not_just_below_price():
    # Regression (DRAM): a Fib level below price but ABOVE the breakdown trigger
    # must NOT be used as the downside target.
    px = 65.0
    lv = {"resistance": 70.0, "support": 57.99}
    fib = {"direction": "up", "swingLow": 50.0, "swingHigh": 75.0,
           "levels": [{"price": 59.76}, {"price": 68.0}],   # 59.76 is below px but ABOVE 57.99
           "extensions": [{"price": 84.0}, {"price": 90.0}], "goldenPocket": {"low": 60, "high": 63}}
    sc = _scenarios(px, lv, fib, [])
    assert sc["bear"]["trigger"] == 57.99
    assert sc["bear"]["target"] != 59.76                 # the nonsensical value is gone
    assert sc["bear"]["target"] < sc["bear"]["trigger"]  # downside target sits BELOW the trigger
    assert sc["bear"]["target"] == 50.0                  # falls back to the swing low


def test_targets_never_contradict_direction():
    # Whatever the geometry, an upside target is above its trigger and a downside
    # target is below its trigger (or omitted).
    px = 100.0
    lv = {"resistance": 102.0, "support": 99.0}
    fib = {"direction": "up", "swingLow": 98.0, "swingHigh": 101.0,
           "levels": [{"price": 99.5}, {"price": 100.5}],
           "extensions": [{"price": 103.0}, {"price": 104.0}], "goldenPocket": {"low": 99.2, "high": 99.8}}
    sc = _scenarios(px, lv, fib, [])
    if sc.get("bull") and sc["bull"].get("target") is not None:
        assert sc["bull"]["target"] > sc["bull"]["trigger"]
    if sc.get("bear") and sc["bear"].get("target") is not None:
        assert sc["bear"]["target"] < sc["bear"]["trigger"]
