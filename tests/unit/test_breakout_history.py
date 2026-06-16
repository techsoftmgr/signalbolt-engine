"""Regression test — build_all_scorecards must fetch PER BUCKET so a high-volume bucket
can't truncate a low-volume one (the turnaround/peak-show-0 bug)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import engine.breakout_history as bh
import engine.alpaca_client as ac


class _FakeQuery:
    def __init__(self, store):
        self.store = store
        self._bucket = None

    def select(self, *a, **k): return self
    def eq(self, col, val):
        if col == "bucket":
            self._bucket = val
        return self
    def gte(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self

    def execute(self):
        class _R: pass
        r = _R()
        r.data = list(self.store.get(self._bucket, []))
        return r


class _FakeSb:
    def __init__(self, store): self.store = store
    def table(self, _name): return _FakeQuery(self.store)


def test_all_scorecards_fetches_per_bucket_no_truncation(monkeypatch):
    # No bars → every episode grades to "open" (outcome None), but totals still populate,
    # which is all we need to prove the per-bucket fetch keeps low-volume buckets.
    monkeypatch.setattr(ac, "get_multi_bars", lambda *a, **k: {}, raising=False)
    store = {
        "vwapReclaim": [{"ticker": "WMT", "bucket": "vwapReclaim", "session_date": "2026-06-15",
                         "entered_at": "2026-06-15T18:00:00Z"}] * 900,   # high-volume bucket
        "turnaround":  [{"ticker": "REGN", "bucket": "turnaround", "session_date": "2026-06-01",
                         "entered_at": "2026-06-01T13:31:00Z"}],          # low-volume, OLD
        "peak":        [{"ticker": "PANW", "bucket": "peak", "session_date": "2026-06-01",
                         "entered_at": "2026-06-01T13:31:00Z"}],
    }
    res = bh.build_all_scorecards(_FakeSb(store), days=90)
    totals = {b["bucket"]: b["scorecard"]["total"] for b in res["buckets"]}
    # All ten buckets present; the low-volume cycle buckets are NOT crowded out.
    assert totals.get("turnaround") == 1
    assert totals.get("peak") == 1
    assert totals.get("vwapReclaim") == 900
    assert {"breakouts", "breakdowns", "topMomentum", "pullbacks", "highVolumeUp",
            "highVolumeDown", "vwapReclaim", "oversoldBounce", "turnaround", "peak"} <= set(totals)
