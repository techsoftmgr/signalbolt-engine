"""Unit tests — paper_trading pure helpers (proposal sizing + realized P&L)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import paper_trading as pt


def test_build_proposal_long():
    sig = {"id": "s1", "ticker": "AAPL", "direction": "LONG", "entry_price": 200.0,
           "stop_loss": 196.0, "target_one": 210.0, "strategy_type": "day_trade",
           "score_breakdown": {"detector_source": "BREAKOUT"}}
    p = pt.build_proposal(sig, alloc=2000)
    assert p["ticker"] == "AAPL" and p["direction"] == "LONG"
    assert p["qty"] == 10                 # floor(2000/200)
    assert p["entry_price"] == 200.0 and p["stop_loss"] == 196.0 and p["target_one"] == 210.0
    assert p["detector_source"] == "BREAKOUT" and p["status"] == "proposed"


def test_build_proposal_min_one_share_and_short():
    sig = {"id": "s2", "ticker": "NVDA", "direction": "short", "entry_price": 5000.0,
           "stop_loss": 5100.0, "target_one": 4800.0}
    p = pt.build_proposal(sig, alloc=2000)
    assert p["qty"] == 1                  # alloc < price → floor to 1, not 0
    assert p["direction"] == "SHORT"


def test_build_proposal_rejects_bad_entry():
    assert pt.build_proposal({"id": "x", "ticker": "X", "entry_price": 0}) is None
    assert pt.build_proposal({"id": "x", "ticker": "X"}) is None


def test_realized_long_and_short():
    pnl, pct = pt.realized("LONG", 100.0, 110.0)
    assert pnl == 10.0 and round(pct, 2) == 10.0
    pnl, pct = pt.realized("SHORT", 100.0, 90.0)          # short profits when price falls
    assert pnl == 10.0 and round(pct, 2) == 10.0
    pnl, pct = pt.realized("LONG", 100.0, 95.0)
    assert pnl == -5.0 and round(pct, 2) == -5.0


def test_realized_guards():
    assert pt.realized("LONG", 0, 100) == (None, None)
    assert pt.realized("LONG", None, 100) == (None, None)


def _closed(det, pct, pnl):
    return {"status": "closed", "detector_source": det, "realized_pct": pct, "realized_pnl": pnl}


def test_scorecard_groups_and_expectancy():
    rows = [
        _closed("BREAKOUT", 5.0, 50), _closed("BREAKOUT", -2.0, -20), _closed("BREAKOUT", 3.0, 30),
        _closed("MOMENTUM_SURGE", -1.0, -10),
        {"status": "filled", "detector_source": "BREAKOUT", "realized_pct": None},   # not closed → ignored
    ]
    sc = pt.scorecard(rows)
    bo = next(g for g in sc["by_detector"] if g["group"] == "BREAKOUT")
    assert bo["trades"] == 3 and bo["wins"] == 2 and bo["losses"] == 1
    assert bo["win_rate"] == round(2 / 3 * 100, 1)
    assert bo["avg_win_pct"] == 4.0 and bo["avg_loss_pct"] == -2.0
    assert bo["expectancy_pct"] == 2.0      # (5 - 2 + 3)/3
    assert bo["total_pnl"] == 60.0
    # ranked by trade count → BREAKOUT (3) before MOMENTUM_SURGE (1)
    assert sc["by_detector"][0]["group"] == "BREAKOUT"
    assert sc["overall"]["trades"] == 4 and sc["overall"]["expectancy_pct"] == round((5 - 2 + 3 - 1) / 4, 2)


def test_scorecard_falls_back_to_strategy_then_unknown():
    rows = [{"status": "closed", "strategy_type": "swing_trade", "realized_pct": 1.0, "realized_pnl": 10},
            {"status": "closed", "realized_pct": 2.0, "realized_pnl": 20}]
    groups = {g["group"] for g in pt.scorecard(rows)["by_detector"]}
    assert groups == {"swing_trade", "unknown"}


def test_scorecard_empty():
    sc = pt.scorecard([])
    assert sc["by_detector"] == [] and sc["overall"]["trades"] == 0
