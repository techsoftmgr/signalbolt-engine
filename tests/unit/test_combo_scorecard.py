"""Unit tests — signal-combination scorecard. Offline, additive."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import combo_scorecard as cs


def test_agg_basic():
    rows = [
        {"result_pct": 2.0, "score_breakdown": {"mfe_pct": 3.0}},
        {"result_pct": -1.0, "score_breakdown": {"mfe_pct": 1.0}},
        {"result_pct": 4.0, "score_breakdown": {}},
    ]
    a = cs._agg(rows)
    assert a["n"] == 3
    assert a["win_pct"] == 67           # 2 of 3 positive
    assert a["avg_pnl"] == round((2 - 1 + 4) / 3, 2)
    assert a["avg_mfe"] == round((3.0 + 1.0) / 2, 2)
    assert a["thin"] is True            # 3 < 30


def test_agg_empty():
    assert cs._agg([]) == {"n": 0}
    assert cs._agg([{"result_pct": None}]) == {"n": 0}


def test_vol_bucket():
    assert cs._vol_bucket(0.8) == "<1.0"
    assert cs._vol_bucket(1.2) == "1.0-1.5"
    assert cs._vol_bucket(1.9) == "1.5-2.0"
    assert cs._vol_bucket(2.5) == ">=2.0"
    assert cs._vol_bucket(None) is None
    assert cs._vol_bucket("x") is None


# ---- scorecard end-to-end with a fake supabase ----
class _Q:
    def __init__(self, data): self._d = data
    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def gte(self, *a, **k): return self
    def in_(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def execute(self): return type("R", (), {"data": self._d})()


class _SB:
    def __init__(self, signals, events): self._s = signals; self._e = events
    def table(self, name): return _Q(self._s if name == "signals" else self._e)


def test_scorecard_segments():
    signals = [
        # two breakdowns, different volume buckets
        {"id": "a", "direction": "SHORT", "strategy_type": "breakdown", "entry_price": 100.0,
         "result_pct": -2.0, "status": "closed", "created_at": "2026-06-01T00:00:00+00:00",
         "score_breakdown": {"relativeVolume": 1.2, "mfe_pct": 1.0, "detector_source": "BREAKDOWN"}},
        {"id": "b", "direction": "SHORT", "strategy_type": "breakdown", "entry_price": 100.0,
         "result_pct": 3.0, "status": "closed", "created_at": "2026-06-02T00:00:00+00:00",
         "score_breakdown": {"relativeVolume": 2.4, "mfe_pct": 4.0, "detector_source": "BREAKDOWN"}},
        # a reversal near its ma20
        {"id": "c", "direction": "LONG", "strategy_type": "turn_forming", "entry_price": 50.0,
         "result_pct": 1.0, "status": "closed", "created_at": "2026-06-03T00:00:00+00:00",
         "score_breakdown": {"ma20": 50.2, "mfe_pct": 2.0}},
    ]
    # 'a' had a near_stop warning fire on it
    events = [{"signal_id": "a", "event_type": "near_stop"},
              {"signal_id": "a", "event_type": "in_profit"}]  # non-warning ignored
    out = cs.scorecard(_SB(signals, events), days=120)
    assert out["available"] is True and out["scored"] == 3
    strategies = {r["strategy"] for r in out["per_strategy"]}
    assert "breakdown" in strategies and "turn_forming" in strategies
    # each strategy carries a nested by-detector-source breakdown
    bd = next(r for r in out["per_strategy"] if r["strategy"] == "breakdown")
    assert any(d["detector"] == "BREAKDOWN" and d["n"] == 2 for d in bd["detectors"])
    vbuckets = {r["bucket"] for r in out["volume"]}
    assert "1.0-1.5" in vbuckets and ">=2.0" in vbuckets
    locb = {r["bucket"] for r in out["location"]}
    assert "near (<1%)" in locb
    # exit-stack: 'a' has 1 warning, b & c have 0
    counts = {r["warnings"]: r["n"] for r in out["exit_stack"]}
    assert counts.get(0) == 2 and counts.get(1) == 1
    # concentration: a/b/c fired on 3 different days -> all solo cohorts
    cbuckets = {r["bucket"]: r["n"] for r in out["concentration"]}
    assert cbuckets.get("solo (1)") == 3


def test_concentration_buckets_basket():
    # 12 LONGs all fired the SAME day -> a "basket (10+)" cohort
    signals = [
        {"id": f"x{i}", "direction": "LONG", "strategy_type": "swing_trade", "entry_price": 100.0,
         "result_pct": -2.0, "status": "closed", "created_at": "2026-05-29T14:00:00+00:00",
         "score_breakdown": {}}
        for i in range(12)
    ]
    out = cs.scorecard(_SB(signals, []), days=120)
    cbuckets = {r["bucket"]: r["n"] for r in out["concentration"]}
    assert cbuckets.get("basket (10+)") == 12


def test_advisor_exit_lock_vs_hold():
    # SHORT @100, held to -2%. First MACD-crossover warning fired at 95 (= +5% if booked).
    signals = [
        {"id": "a", "direction": "SHORT", "strategy_type": "breakdown", "entry_price": 100.0,
         "result_pct": -2.0, "status": "closed", "created_at": "2026-06-01T00:00:00+00:00",
         "score_breakdown": {}},
    ]
    events = [
        {"signal_id": "a", "event_type": "advisor_momentum", "price": 95.0, "created_at": "2026-06-01T15:00:00+00:00"},
        {"signal_id": "a", "event_type": "advisor_momentum", "price": 96.0, "created_at": "2026-06-01T15:05:00+00:00"},  # later → ignored
    ]
    ax = cs.scorecard(_SB(signals, events), days=120)["advisor_exit"]
    assert ax["n"] == 1
    assert ax["avg_lock_pnl"] == 5.0 and ax["avg_hold_pnl"] == -2.0   # book +5% vs hold -2%
    assert ax["edge"] == 7.0 and ax["lock_beat_hold_pct"] == 100


def test_scorecard_no_sb():
    assert cs.scorecard(None)["available"] is False
