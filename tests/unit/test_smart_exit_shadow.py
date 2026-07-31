"""Shadow backfill glue: only shadow-tagged closed trades are replayed; the alternate
outcome + delta-vs-actual is written to score_breakdown; live/untagged rows untouched."""
import types
from datetime import datetime, timezone, timedelta

import pandas as pd

from engine import smart_exit_shadow as sxs
from engine import exit_engine, alpaca_client


class _Q:
    def __init__(self, store):
        self.store = store; self.up = None; self.idv = None
    def select(self, *a, **k): return self
    def eq(self, col, val):
        if self.up is not None:
            self.idv = val
        return self
    def gte(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def order(self, *a, **k): return self
    def range(self, *a, **k): return self
    def update(self, payload): self.up = payload; return self
    def execute(self):
        if self.up is not None:
            for r in self.store:
                if r["id"] == self.idv:
                    r["score_breakdown"] = self.up["score_breakdown"]
            return types.SimpleNamespace(data=[])
        return types.SimpleNamespace(data=list(self.store))


class _SB:
    def __init__(self, store): self.store = store
    def table(self, name): return _Q(self.store)


def test_backfill_writes_final_only_for_shadow(monkeypatch):
    ed = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    store = [
        {"id": "1", "ticker": "AAA", "direction": "LONG", "entry_price": 100.0, "stop_loss": 90.0,
         "created_at": ed, "result_pct": 1.5,
         "score_breakdown": {"smart_exit_managed": True, "smart_exit_mode": "shadow"}},
        # live-mode row → not a shadow candidate → must be left untouched
        {"id": "2", "ticker": "BBB", "direction": "LONG", "entry_price": 50.0, "stop_loss": 45.0,
         "created_at": ed, "result_pct": 2.0,
         "score_breakdown": {"smart_exit_managed": True, "smart_exit_mode": "live"}},
    ]
    monkeypatch.setattr(alpaca_client, "get_bars",
                        lambda *a, **k: pd.DataFrame({"high": [1.0] * 50, "low": [1.0] * 50, "close": [1.0] * 50}))
    monkeypatch.setattr(exit_engine, "replay",
                        lambda *a, **k: {"pnl_pct": 8.0, "exit_reason": "giveback_cap",
                                         "held_days": 6, "exit_price": 108.0})

    res = sxs.backfill_batch(_SB(store), limit=10)

    assert res["evaluated"] == 1 and res["smart_better"] == 1        # only the shadow row; 8.0 > 1.5
    fin = store[0]["score_breakdown"]["smart_exit_shadow_final"]
    assert fin["pnl_pct"] == 8.0 and fin["actual_pct"] == 1.5 and fin["delta"] == 6.5
    assert "smart_exit_shadow_final" not in store[1]["score_breakdown"]  # live row untouched


def test_backfill_skips_premature_window_end(monkeypatch):
    # a RECENT shadow trade whose replay hasn't resolved (window_end) must stay pending
    ed = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    store = [{"id": "1", "ticker": "AAA", "direction": "LONG", "entry_price": 100.0, "stop_loss": 90.0,
              "created_at": ed, "result_pct": 0.5,
              "score_breakdown": {"smart_exit_managed": True, "smart_exit_mode": "shadow"}}]
    monkeypatch.setattr(alpaca_client, "get_bars",
                        lambda *a, **k: pd.DataFrame({"high": [1.0] * 50, "low": [1.0] * 50, "close": [1.0] * 50}))
    monkeypatch.setattr(exit_engine, "replay",
                        lambda *a, **k: {"pnl_pct": 0.0, "exit_reason": "window_end",
                                         "held_days": 5, "exit_price": 100.0})
    res = sxs.backfill_batch(_SB(store), limit=10)
    assert res["evaluated"] == 0 and res["pending"] == 1
    assert "smart_exit_shadow_final" not in store[0]["score_breakdown"]


def _final(det, mode, actual, shadow, reason):
    return {"direction": "LONG", "result_pct": actual, "status": "closed",
            "score_breakdown": {"detector_source": det, "smart_exit_managed": True, "smart_exit_mode": mode,
                                "smart_exit_shadow_final": {"pnl_pct": shadow, "exit_reason": reason}}}


def test_scorecard_aggregates_by_detector():
    store = [
        _final("TREND_MOMENTUM", "shadow", -3.0, 5.0, "giveback_cap"),   # smart better by +8
        _final("PEAK_FORMING", "shadow", 2.0, 1.0, "confluence_break"),  # smart worse by -1
        {"direction": "LONG", "result_pct": 1.0, "status": "closed",     # no final → ignored
         "score_breakdown": {"detector_source": "SMC"}},
    ]
    res = sxs.scorecard(_SB(store), days=30)
    assert res["evaluated"] == 2 and res["overall"]["n"] == 2
    assert res["overall"]["delta_total"] == 7.0        # shadow 6 − actual (−1)
    assert res["overall"]["shadow_better"] == 1 and res["overall"]["shadow_worse"] == 1
    assert res["detectors"][0]["detector"] == "TREND_MOMENTUM"   # sorted by delta desc
    assert res["detectors"][0]["delta_total"] == 8.0
