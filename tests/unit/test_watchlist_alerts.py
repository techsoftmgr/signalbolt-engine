"""watchlist_alerts._events — transition → alert mapping (pure, no DB)."""
from engine import watchlist_alerts as wa


def _base(**over):
    s = {"turn": "none", "peak": "none", "status": "", "aboveMA": True, "cmf": 0.0}
    s.update(over)
    return s


def _keys(prev, cur):
    return {e[0] for e in wa._events(prev, cur)}


def test_cmf_bullish_cross_fires():
    assert "cmf_bull" in _keys(_base(cmf=-0.06), _base(cmf=0.08))


def test_cmf_bearish_cross_fires():
    assert "cmf_bear" in _keys(_base(cmf=0.04), _base(cmf=-0.07))


def test_cmf_no_cross_when_within_buffer():
    # ends positive but only +0.02 (< 0.05 buffer) → no alert
    assert "cmf_bull" not in _keys(_base(cmf=-0.03), _base(cmf=0.02))
    # already positive, no fresh cross
    assert "cmf_bull" not in _keys(_base(cmf=0.06), _base(cmf=0.10))


def test_cmf_none_values_safe():
    assert _keys(_base(cmf=None), _base(cmf=0.08)) == set()   # no prev flow → no cross


def test_existing_events_still_fire():
    assert "buyzone" in _keys(_base(turn="none"), _base(turn="buyzone"))
    assert "topping" in _keys(_base(peak="none"), _base(peak="watch"))
