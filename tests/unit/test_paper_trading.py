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
