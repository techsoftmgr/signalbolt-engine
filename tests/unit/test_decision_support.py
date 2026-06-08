"""Unit tests — Decision Support layer (pure derivation). Additive.

Covers the 10 required scenarios: action logic (WAIT / BUY ZONE / WATCH / AVOID),
probabilities always total 100, missing fib / support-resistance / historical
degrade gracefully, and the derivation never mutates the existing read.
"""
import copy
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import decision_support as ds


def _base(**over):
    """A complete, neutral-ish chart_read.analyze()-shaped read; override per test."""
    r = {
        "ticker": "TST", "price": 100.0, "bias": "neutral",
        "taBias": "neutral", "quantBias": "neutral", "agreement": "agree",
        "shortTerm": "neutral",
        "trend": {"d1": "sideways", "h1": "sideways", "m15": "sideways"},
        "mtf": {"state": "mixed", "dir": "none"},
        "levels": {"resistance": 110.0, "support": 90.0},
        "trendlines": {"vsSupport": "above", "vsResistance": "testing"},
        "fib": {"direction": "up", "goldenPocket": {"low": 84.0, "high": 88.0},
                "invalidation": 80.0, "target": 130.0},
        "volumeRegime": "normal", "confidence": 60,
        "scenarios": {"bull": {"trigger": 110.0, "then": "up", "target": 120.0},
                      "bear": {"trigger": 90.0, "then": "down", "target": 82.0}},
        "narrative": ["bullet one", "bullet two"],
    }
    r.update(over)
    return r


# 1 ── quant bullish + technical bullish + near resistance => WAIT
def test_action_wait_near_resistance():
    r = _base(taBias="bullish", quantBias="bullish", agreement="agree",
              trend={"d1": "up", "h1": "up", "m15": "up"},
              levels={"resistance": 101.0, "support": 90.0})   # 1% to resistance = near
    out = ds.derive(r)
    assert out["action"] == "WAIT"
    assert "resistance" in out["reason"].lower()


# 2 ── quant bullish + technical bullish + price in golden pocket => BUY ZONE
def test_action_buy_zone_golden_pocket():
    r = _base(price=86.0, taBias="bullish", quantBias="bullish", agreement="agree",
              trend={"d1": "up", "h1": "up", "m15": "up"},
              levels={"resistance": 96.0, "support": 80.0},      # resistance ~12% away (not near)
              fib={"direction": "up", "goldenPocket": {"low": 84.0, "high": 88.0},
                   "invalidation": 78.0, "target": 120.0})
    out = ds.derive(r)
    assert out["action"] == "BUY ZONE"


# 3 ── quant/technical disagreement => WATCH
def test_action_watch_on_disagreement():
    r = _base(taBias="bullish", quantBias="bearish", agreement="disagree",
              trend={"d1": "sideways", "h1": "down", "m15": "up"},
              levels={"resistance": 112.0, "support": 88.0})
    out = ds.derive(r)
    assert out["action"] == "WATCH"


# 4 ── price loses support => AVOID
def test_action_avoid_on_lost_support():
    r = _base(taBias="bearish", quantBias="bearish", agreement="agree",
              trend={"d1": "down", "h1": "down", "m15": "down"},
              trendlines={"vsSupport": "below", "vsResistance": "below"})
    out = ds.derive(r)
    assert out["action"] == "AVOID"


# 5 ── probabilities always total exactly 100
def test_probabilities_total_100():
    reads = [
        _base(taBias="bullish", quantBias="bullish", trend={"d1": "up", "h1": "up", "m15": "up"}),
        _base(taBias="bearish", quantBias="bearish", trend={"d1": "down", "h1": "down", "m15": "down"}),
        _base(taBias="bullish", quantBias="bearish", agreement="disagree"),
        _base(),  # neutral
        _base(quantBias=None, agreement="n/a"),
    ]
    for r in reads:
        out = ds.derive(r)
        total = out["bullish_probability"] + out["neutral_probability"] + out["bearish_probability"]
        assert total == 100, f"got {total} for {r['taBias']}/{r['quantBias']}"
        for k in ("bullish_probability", "neutral_probability", "bearish_probability"):
            assert 0 <= out[k] <= 100


# 6 ── missing Fibonacci data does not crash
def test_missing_fib_ok():
    r = _base(taBias="bullish", quantBias="bullish")
    r.pop("fib")
    out = ds.derive(r)
    assert out["available"] is True
    assert "bullish_probability" in out
    assert out["entry_quality"]["ideal_pullback_zone"] is None  # no fib → no pullback zone


# 7 ── missing support/resistance data does not crash
def test_missing_levels_ok():
    r = _base(taBias="bullish", quantBias="bullish")
    r["levels"] = {}
    r.pop("scenarios")
    out = ds.derive(r)
    assert out["available"] is True
    assert isinstance(out["scenario_tree"], dict)
    assert isinstance(out["reasons_for"], list)


# 8 ── missing historical data shows the unavailable state
def test_historical_unavailable():
    out = ds.derive(_base(taBias="bullish"))          # no historical passed
    assert out["historical_similar_setups"]["available"] is False
    # And the DB helper with no client also degrades cleanly:
    assert ds.historical_similar_setups(None, "TST", "bullish")["available"] is False
    assert ds.historical_similar_setups(object(), "TST", "neutral")["available"] is False


def test_historical_uses_real_rows_when_enough():
    # Minimal fluent stub mirroring supabase-py's chain, incl. the `.not_.is_(...)`
    # null-filter (not_ is accessed as an attribute, then is_() is called).
    class _Q:
        _rows = [{"forward_return_pct": v, "horizon_days": 5,
                  "created_at": f"2026-05-{10 + i:02d}T00:00:00Z"}
                 for i, v in enumerate([3.0, -1.5, 2.2, 4.1, -0.5, 1.8, 5.0, -2.0, 0.9, 3.3])]
        def select(self, *a, **k): return self
        def eq(self, *a, **k): return self
        @property
        def not_(self): return self
        def is_(self, *a, **k): return self
        def order(self, *a, **k): return self
        def limit(self, *a, **k): return self
        def execute(self): return type("R", (), {"data": self._rows})()
    class _SB:
        def table(self, *_a, **_k): return _Q()
    out = ds.historical_similar_setups(_SB(), "TST", "bullish")
    assert out["available"] is True
    assert out["count"] == 10
    assert 0 <= out["win_rate"] <= 100
    assert out["avg_gain"] is not None and out["avg_loss"] is not None
    assert out["horizon_days"] == 5


# 9 ── derive does NOT mutate the existing read (existing fields preserved)
def test_derive_does_not_mutate_read():
    r = _base(taBias="bullish", quantBias="bullish")
    snapshot = copy.deepcopy(r)
    _ = ds.derive(r)
    assert r == snapshot   # every original field intact, nothing added/removed/changed


# 10 ── absent/empty read degrades to an unavailable object (UI null-check path)
def test_empty_read_graceful():
    assert ds.derive({})["available"] is False
    assert ds.derive(None)["available"] is False
    out = ds.derive({"ticker": "X"})   # no price
    assert out["available"] is False
    assert "disclaimer" in out


# extra ── structural completeness of a full derive
def test_scorecard_present_and_consistent():
    out = ds.derive(_base(taBias="bullish", quantBias="bullish",
                          trend={"d1": "up", "h1": "up", "m15": "up"}))
    sc = out["scorecard"]
    assert sc["total"] == len(sc["items"])
    assert sc["bullish"] + sc["bearish"] == sc["total"]
    assert all(("label" in i and "supportive" in i) for i in sc["items"])
    assert "condition" in sc["summary"].lower()


def test_action_reasons_are_not_predictive():
    # reasons describe STATE, not a forecast — no "favors"/"will"/"should"
    for r in (_base(taBias="bullish", quantBias="bullish", trend={"d1": "up", "h1": "up", "m15": "up"},
                    levels={"resistance": 101.0, "support": 90.0}),
              _base(taBias="bearish", quantBias="bearish", trend={"d1": "down", "h1": "down", "m15": "down"},
                    trendlines={"vsSupport": "below", "vsResistance": "below"})):
        reason = ds.derive(r)["reason"].lower()
        assert "favors" not in reason and "will " not in reason


def test_full_structure_present():
    out = ds.derive(_base(taBias="bullish", quantBias="bullish",
                          trend={"d1": "up", "h1": "up", "m15": "up"}))
    for key in ("action", "reason", "confidence", "trade_quality", "risk_reward_quality",
                "bullish_probability", "scenario_tree", "entry_quality", "reasons_for",
                "reasons_against", "risk_meter", "plain_english_read",
                "historical_similar_setups", "signal_freshness", "tags", "disclaimer"):
        assert key in out, f"missing {key}"
    assert out["confidence"] in ("Low", "Medium", "High")
    assert out["trade_quality"] in ("A+", "A", "B", "C", "D")
    assert out["risk_reward_quality"] in ("Poor", "Fair", "Good", "Excellent")
    assert out["risk_meter"]["level"] in ("Low", "Medium", "High")
    assert out["plain_english_read"].endswith(".")
    assert "not financial advice" in out["disclaimer"].lower()
