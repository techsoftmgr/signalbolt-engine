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
