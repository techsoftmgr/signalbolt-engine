"""Unit tests — Expert Read self-grade (read_accuracy). Additive, offline.

The factual grader `_grade_levels` is pure (bars in → tested/held flags out), so
it's exercised directly. The log/score entrypoints are checked for the no-db and
never-raise guarantees.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import read_accuracy as ra


def _bars(ohlc):
    return [{"high": h, "low": l, "close": c} for (h, l, c) in ohlc]


# ── support ───────────────────────────────────────────────────────────────────
def test_support_held_when_tested_and_not_broken():
    # support flagged at 100; price dips to 100.2 (tested) and never closes below 99
    g = ra._grade_levels(100.0, None, _bars([(103, 100.2, 102), (104, 101, 103)]))
    assert g["support_tested"] is True
    assert g["support_held"] is True


def test_support_broken_when_closed_through():
    # tested at 100.3, then a daily close at 98 (>1% below 100) → broke
    g = ra._grade_levels(100.0, None, _bars([(101, 100.3, 100.5), (100, 97, 98.0)]))
    assert g["support_tested"] is True
    assert g["support_held"] is False


def test_support_untested_stays_none():
    # price never comes within 0.5% of 100 (lowest low 105) → not tested, held undefined
    g = ra._grade_levels(100.0, None, _bars([(110, 105, 108), (112, 106, 110)]))
    assert g["support_tested"] is False
    assert g["support_held"] is None


# ── resistance ──────────────────────────────────────────────────────────────────
def test_resistance_held_when_tested_and_not_broken():
    # resistance at 100; price pokes 99.8 (within 0.5%) but no close above 101
    g = ra._grade_levels(None, 100.0, _bars([(99.8, 97, 98), (100.4, 96, 97)]))
    assert g["resistance_tested"] is True
    assert g["resistance_held"] is True


def test_resistance_broken_when_closed_above():
    g = ra._grade_levels(None, 100.0, _bars([(100.2, 98, 99), (105, 100, 103.0)]))
    assert g["resistance_tested"] is True
    assert g["resistance_held"] is False


def test_both_levels_and_empty_bars():
    g = ra._grade_levels(90.0, 110.0, [])
    assert g == {"support_tested": None, "support_held": None,
                 "resistance_tested": None, "resistance_held": None}


def test_grade_never_raises_on_bad_input():
    g = ra._grade_levels("x", None, [{"high": "nan", "low": None, "close": 1}])
    assert isinstance(g, dict)


# ── entrypoints: no-db + never-raise ────────────────────────────────────────────
def test_log_and_score_no_sb():
    assert ra.log_levels(None)["logged"] == 0
    assert ra.score_levels(None)["scored"] == 0


def test_stats_no_sb():
    assert ra.stats(None)["available"] is False


def test_stats_aggregates(monkeypatch):
    rows = [
        {"support_tested": True, "support_held": True, "resistance_tested": False, "resistance_held": None},
        {"support_tested": True, "support_held": False, "resistance_tested": True, "resistance_held": True},
        {"support_tested": False, "support_held": None, "resistance_tested": True, "resistance_held": True},
    ]

    class _Q:
        def select(self, *a, **k): return self
        def gte(self, *a, **k): return self
        def not_(self): return self
        def is_(self, *a, **k): return self
        def limit(self, *a, **k): return self
        def execute(self): return type("R", (), {"data": rows})()
    # not_ is a property-like chain in supabase-py; emulate .not_.is_(...)
    q = _Q()
    q.not_ = q

    class _SB:
        def table(self, *a, **k): return q
    out = ra.stats(_SB())
    assert out["available"] is True and out["scored"] == 3
    assert out["support"]["tested"] == 2 and out["support"]["held_pct"] == 50
    assert out["resistance"]["tested"] == 2 and out["resistance"]["held_pct"] == 100


def test_stats_cached_caches_and_serves(monkeypatch):
    class _KV:
        def __init__(self): self.d = {}
        def get_json(self, k): return self.d.get(k)
        def set_json(self, k, v, ttl): self.d[k] = v
    import engine.cache as cache_mod
    monkeypatch.setattr(cache_mod, "kv", _KV())
    calls = {"n": 0}

    def _fake_stats(sb, days=90):
        calls["n"] += 1
        return {"available": True, "scored": 5, "support": {"tested": 5, "held_pct": 80},
                "resistance": {"tested": 3, "held_pct": 67}, "note": "x"}
    monkeypatch.setattr(ra, "stats", _fake_stats)
    a = ra.stats_cached(object())   # computes + caches
    b = ra.stats_cached(object())   # served from cache
    assert a == b and calls["n"] == 1


def test_stats_cached_does_not_cache_unavailable(monkeypatch):
    class _KV:
        def __init__(self): self.d = {}
        def get_json(self, k): return self.d.get(k)
        def set_json(self, k, v, ttl): self.d[k] = v
    import engine.cache as cache_mod
    monkeypatch.setattr(cache_mod, "kv", _KV())
    monkeypatch.setattr(ra, "stats", lambda sb, days=90: {"available": False})
    ra.stats_cached(object())
    assert cache_mod.kv.d == {}     # unavailable result is not cached (retries next call)
