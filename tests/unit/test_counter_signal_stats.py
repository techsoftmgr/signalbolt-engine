"""Unit tests — counter-signal exit scorer (lock-vs-hold). Additive, offline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import counter_signal_stats as css


def test_lock_pnl_short_and_long():
    # SHORT: locking lower than entry = profit
    assert css._lock_pnl("SHORT", 100.0, 90.0) == 10.0
    # LONG: locking higher than entry = profit
    assert css._lock_pnl("LONG", 100.0, 110.0) == 10.0
    assert css._lock_pnl("SHORT", 0, 90) is None
    assert css._lock_pnl("LONG", "x", 1) is None


def test_stage_of():
    assert css._stage_of("Counter-signal: turnaround (confirmed) ...") == "confirmed"
    assert css._stage_of("turnaround (forming) vs this short") == "forming"
    assert css._stage_of("something else") == "unknown"


def test_aggregate_lock_beats_hold():
    # 3 closed: locking captured more than holding in 2 of 3
    rows = [
        {"lock_pnl": 2.8, "hold_pnl": -1.0, "stage": "forming"},   # lock won (short reversed)
        {"lock_pnl": 3.0, "hold_pnl": 1.0,  "stage": "confirmed"}, # lock won
        {"lock_pnl": 1.0, "hold_pnl": 4.0,  "stage": "confirmed"}, # hold won (kept running)
        {"lock_pnl": 1.5, "hold_pnl": None,  "stage": "forming"},  # still open
    ]
    a = css._aggregate(rows)
    assert a["total_events"] == 4
    assert a["scored"] == 3
    assert a["open_pending"] == 1
    assert a["overall"]["lock_beat_hold_pct"] == 67
    assert a["overall"]["avg_lock_pnl"] == round((2.8 + 3.0 + 1.0) / 3, 2)
    assert a["overall"]["edge"] == round(a["overall"]["avg_lock_pnl"] - a["overall"]["avg_hold_pnl"], 2)
    assert "confirmed" in a["by_stage"] and a["by_stage"]["confirmed"]["n"] == 2


def test_aggregate_empty():
    a = css._aggregate([])
    assert a["scored"] == 0 and a["overall"] == {"n": 0}


def test_stats_no_sb():
    assert css.stats(None)["available"] is False


# ---- backfill (offline, fake supabase) ----
class _Q:
    def __init__(self, data): self._data = data
    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def execute(self): return type("R", (), {"data": self._data})()


class _SB:
    def __init__(self, signals, events): self._s = signals; self._e = events
    def table(self, name): return _Q(self._s if name == "signals" else self._e)


def test_backfill_dryrun_finds_opposing_pair():
    # A filled+closed SHORT on XYZ that lost (-3%); a turn_forming LONG fired while it was open.
    signals = [
        {"id": "pos1", "ticker": "XYZ", "direction": "SHORT", "strategy_type": "breakdown",
         "entry_price": 100.0, "result_pct": -3.0, "status": "closed",
         "created_at": "2026-06-01T15:00:00+00:00", "closed_at": "2026-06-05T20:00:00+00:00"},
        {"id": "opp1", "ticker": "XYZ", "direction": "LONG", "strategy_type": "turn_forming",
         "entry_price": 96.0, "result_pct": 2.0, "status": "closed",
         "created_at": "2026-06-03T15:00:00+00:00", "closed_at": "2026-06-06T20:00:00+00:00"},
    ]
    r = css.backfill(_SB(signals, []), commit=False)
    assert r["pairs"] == 1 and r["committed"] is False
    ev = r["preview"][0]
    assert ev["signal_id"] == "pos1" and ev["event_type"] == "counter_signal"
    # SHORT entered 100, lock at 96 = +4% locked
    assert "+4.0%" in ev["note"] and ev["price"] == 96.0


def test_backfill_skips_already_recorded_and_open():
    signals = [
        {"id": "pos1", "ticker": "XYZ", "direction": "SHORT", "strategy_type": "breakdown",
         "entry_price": 100.0, "result_pct": -3.0, "status": "closed",
         "created_at": "2026-06-01T15:00:00+00:00", "closed_at": "2026-06-05T20:00:00+00:00"},
        {"id": "opp1", "ticker": "XYZ", "direction": "LONG", "strategy_type": "turn_forming",
         "entry_price": 96.0, "result_pct": 2.0, "status": "closed",
         "created_at": "2026-06-03T15:00:00+00:00", "closed_at": "2026-06-06T20:00:00+00:00"},
        # an open (not closed) short — should be ignored (no hold outcome)
        {"id": "pos2", "ticker": "XYZ", "direction": "SHORT", "strategy_type": "breakdown",
         "entry_price": 100.0, "result_pct": None, "status": "active",
         "created_at": "2026-06-04T15:00:00+00:00", "closed_at": None},
    ]
    # pos1 already has a counter_signal event -> skip it; pos2 is open -> skip
    r = css.backfill(_SB(signals, [{"signal_id": "pos1"}]), commit=False)
    assert r["pairs"] == 0
