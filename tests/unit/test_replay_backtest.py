"""Unit tests — replay_backtest.replay (pure no-look-ahead bar-walk). bars are
(high, low, close). cost_pct=0 here to test gross outcomes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine.replay_backtest import replay

C0 = {"cost_pct": 0.0}


def test_long_hits_target():
    # entry 100, target +6% (106), stop -3% (97). Bar 2 highs to 107.
    bars = [(101, 99, 100), (104, 100, 103), (107, 105, 106)]
    r = replay("LONG", 100, bars, {"stop_pct": 3, "target_pct": 6, **C0})
    assert r["outcome"] == "win" and r["exit_reason"] == "target"
    assert r["realized_pct"] == 6.0


def test_long_hits_stop():
    bars = [(101, 99, 100), (100, 96, 97)]          # bar 2 low 96 → -4% < -3% stop
    r = replay("LONG", 100, bars, {"stop_pct": 3, "target_pct": 6, **C0})
    assert r["outcome"] == "loss" and r["exit_reason"] == "stop"
    assert r["realized_pct"] == -3.0


def test_same_bar_stop_wins_tie():
    # one bar spans both stop (-3%) and target (+6%): conservative → stop
    bars = [(108, 95, 100)]
    r = replay("LONG", 100, bars, {"stop_pct": 3, "target_pct": 6, **C0})
    assert r["exit_reason"] == "stop_and_target" and r["realized_pct"] == -3.0


def test_no_lookahead_trail_uses_prior_peak():
    # rallies to +8% then a later bar pulls back through the trail.
    # trail 3%: after the +8 bar, stop locks to +5%. Next bar low = +4% → exit +5%.
    bars = [(105, 100, 104), (108, 105, 107), (106.5, 105, 105.5)]
    # bar1 fav +5; bar2 fav +8 → peak 8, trail stop = 5; bar3 low 105 → +5% ≤ stop5 → exit 5
    r = replay("LONG", 100, bars, {"stop_pct": 4, "trail_pct": 3, **C0})
    assert r["exit_reason"] == "stop" and r["realized_pct"] == 5.0


def test_breakeven_protects():
    # +4% then back to entry: breakeven_at 3 raises stop to 0 → exit at breakeven
    bars = [(104, 100, 103), (103, 100, 100)]
    r = replay("LONG", 100, bars, {"stop_pct": 5, "breakeven_at_pct": 3, **C0})
    assert r["exit_reason"] == "stop" and r["realized_pct"] == 0.0


def test_time_stop_exits_at_close():
    bars = [(101, 99, 100.5), (102, 100, 101.0)]
    r = replay("LONG", 100, bars, {"stop_pct": 10, "time_stop_bars": 2, **C0})
    assert r["exit_reason"] == "time" and r["realized_pct"] == 1.0   # close 101


def test_short_direction_mirrors():
    # SHORT entry 100, target +5% favorable = price 95; stop -3% = price 103
    bars = [(101, 99, 100), (100, 94, 95)]          # bar2 low 94 → +6% favorable
    r = replay("SHORT", 100, bars, {"stop_pct": 3, "target_pct": 5, **C0})
    assert r["outcome"] == "win" and r["exit_reason"] == "target" and r["realized_pct"] == 5.0


def test_never_exits_marks_last_close():
    bars = [(101, 99, 100), (102, 100, 101.5)]
    r = replay("LONG", 100, bars, {"stop_pct": 10, "target_pct": 10, **C0})
    assert r["exit_reason"] == "end" and r["realized_pct"] == 1.5


def test_cost_subtracted():
    bars = [(107, 105, 106)]
    r = replay("LONG", 100, bars, {"stop_pct": 3, "target_pct": 6, "cost_pct": 0.1})
    assert r["realized_pct"] == 6.0 and r["realized_net_pct"] == 5.9


def test_macd_lock_exits_on_flip_in_profit():
    # LONG in profit (+5%), MACD histogram flips +→- on bar 2 → lock at that close
    bars = [(104, 100, 103), (106, 104, 105), (105.5, 104, 105)]
    macd = [0.5, 0.4, -0.2]   # positive, positive, flips negative on bar 3
    r = replay("LONG", 100, bars, {"stop_pct": 10, "macd_hist": macd,
                                   "macd_lock_arm": 2.0, "cost_pct": 0.0})
    assert r["exit_reason"] == "macd_lock" and r["realized_pct"] == 5.0


def test_macd_lock_inactive_below_arm():
    # only +1% profit (< arm 2%) → MACD flip ignored, rides to end
    bars = [(101, 100, 100.5), (101.5, 100, 101.0)]
    macd = [0.3, -0.3]
    r = replay("LONG", 100, bars, {"stop_pct": 10, "macd_hist": macd,
                                   "macd_lock_arm": 2.0, "cost_pct": 0.0})
    assert r["exit_reason"] == "end"
