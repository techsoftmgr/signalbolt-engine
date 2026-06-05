"""Unit tests — daily_performance._aggregate builds the EOD snapshot row from
already-fetched closed/active/price/regime inputs (pure)."""
import sys, os, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine.daily_performance import _aggregate

TD = dt.date(2026, 6, 5)

def _closed(tk, dr, pct, conv, det, mfe=None, opened="2026-06-04"):
    sb = {"detector_source": det}
    if mfe is not None: sb["mfe_pct"] = mfe
    return {"ticker": tk, "direction": dr, "result_pct": pct, "confidence_score": conv,
            "score_breakdown": sb, "created_at": opened + "T14:00:00+00:00",
            "closed_at": "2026-06-05T18:00:00+00:00"}

def test_closed_aggregates_and_giveback():
    closed = [
        _closed("CLSK", "SHORT", 12.0, 80, "DISTRIB_FORMING", mfe=14.0),   # gave back 2
        _closed("MU",   "LONG", -12.0, 75, "TREND_MOMENTUM", mfe=11.0),    # gave back 23
        _closed("MRVL", "LONG",  25.0, 75, "TREND_MOMENTUM", mfe=56.0),    # gave back 31
    ]
    r = _aggregate(closed, [], {}, [], TD)
    assert r["closed_n"] == 3 and r["closed_wins"] == 2
    assert round(r["closed_net_pct"], 1) == 25.0
    assert r["long_n"] == 2 and r["short_n"] == 1
    assert r["short_net_pct"] == 12.0
    assert r["giveback_pct"] == 56.0          # 2 + 23 + 31
    assert r["top_winner"]["ticker"] == "MRVL" and r["top_loser"]["ticker"] == "MU"
    assert r["by_detector"]["TREND_MOMENTUM"]["n"] == 2
    assert r["by_conviction"]["A (80-89)"]["n"] == 1 and r["by_conviction"]["B+ (70-79)"]["n"] == 2
    assert r["carried_n"] == 3                 # all opened 06-04, closed 06-05

def test_carried_vs_same_day():
    closed = [_closed("X", "SHORT", 3.0, 60, "BREAKDOWN", opened="2026-06-05")]  # opened+closed today
    r = _aggregate(closed, [], {}, [], TD)
    assert r["carried_n"] == 0

def test_active_book_and_near_levels():
    active = [
        {"ticker": "AAA", "direction": "LONG",  "entry_price": 100, "stop_loss": 95, "target_one": 110,
         "score_breakdown": {"mfe_pct": 5.0}},   # cur 103 → +3% unreal, peak 5 → giveback 2
        {"ticker": "BBB", "direction": "SHORT", "entry_price": 50, "stop_loss": 50.5, "target_one": 45,
         "score_breakdown": {}},                 # cur 50.4 → near stop (within 1.5%)
    ]
    r = _aggregate([], active, {"AAA": 103.0, "BBB": 50.4}, [], TD)
    assert r["active_n"] == 2 and r["active_long_n"] == 1 and r["active_short_n"] == 1
    assert round(r["active_net_unreal_pct"], 1) == 2.2   # +3% long + (-0.8%) short
    assert r["active_near_levels"] == 1                  # BBB near its stop
    assert r["active_giveback_pct"] == 2.0

def test_regime_path():
    rg = [{"captured_at": "2026-06-05T13:00:00Z", "session": "pre", "regime_type": "TRENDING_BULL", "vix": 17},
          {"captured_at": "2026-06-05T16:30:00Z", "session": "rth", "regime_type": "PANIC", "vix": 21}]
    r = _aggregate([], [], {}, rg, TD)
    assert r["regime_close"] == "PANIC" and r["vix"] == 21
    assert "pre TRENDING_BULL > rth PANIC" == r["regime_path"]

def test_empty_is_graceful():
    r = _aggregate([], [], {}, [], TD)
    assert r["closed_n"] == 0 and r["active_n"] == 0 and r["top_winner"] is None
