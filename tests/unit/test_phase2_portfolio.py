"""Unit tests — Phase 2 Portfolio Doctor (pure analyze + CSV adapter) + signal
follow-up status. Additive; existing behavior untouched."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine.phase2 import portfolio_doctor as pd
from engine.phase2 import signal_followup as sf
from engine.phase2.brokers import csv_adapter


def _sectors():
    return {"AAPL": "Technology", "NVDA": "Technology", "MSFT": "Technology",
            "JPM": "Financials", "XOM": "Energy"}


def test_analyze_flags_concentration():
    holdings = [
        {"ticker": "NVDA", "qty": 100, "avg_price": 100, "current_price": 130},  # big tech winner
        {"ticker": "AAPL", "qty": 50, "avg_price": 150, "current_price": 140},
        {"ticker": "MSFT", "qty": 20, "avg_price": 300, "current_price": 320},
    ]
    rep = pd.analyze(holdings, cash=2000, sectors=_sectors())
    assert 0 <= rep["score"] <= 100
    # ~100% technology → sector concentration flagged
    assert any("Technology" in r for r in rep["risks"])
    assert rep["largest_winner"]["ticker"] == "NVDA"
    assert rep["sector_allocation"]["Technology"] > 50


def test_analyze_balanced_scores_higher():
    bal = [
        {"ticker": "AAPL", "qty": 10, "avg_price": 100, "current_price": 100},
        {"ticker": "JPM", "qty": 10, "avg_price": 100, "current_price": 100},
        {"ticker": "XOM", "qty": 10, "avg_price": 100, "current_price": 100},
        {"ticker": "MSFT", "qty": 10, "avg_price": 100, "current_price": 100},
        {"ticker": "NVDA", "qty": 10, "avg_price": 100, "current_price": 100},
    ]
    conc = [{"ticker": "NVDA", "qty": 100, "avg_price": 100, "current_price": 100}]
    assert pd.analyze(bal, cash=600, sectors=_sectors())["score"] > \
           pd.analyze(conc, cash=0, sectors=_sectors())["score"]


def test_coach_is_educational():
    rep = pd.analyze([{"ticker": "NVDA", "qty": 100, "avg_price": 100, "current_price": 100}],
                     cash=0, sectors=_sectors())
    txt = pd.coach(rep)
    assert "not financial advice" in txt.lower() and "health score" in txt.lower()


def test_csv_adapter_parses_messy_export():
    csv = "Symbol,Quantity,Avg Cost,Last Price\nAAPL,10,150.00,$170.50\nNVDA,5,\"1,000\",1200\nCASH,,,5000\n"
    holdings, cash = csv_adapter.parse(csv)
    by = {h["ticker"]: h for h in holdings}
    assert by["AAPL"]["qty"] == 10 and by["AAPL"]["current_price"] == 170.50
    assert by["NVDA"]["avg_price"] == 1000.0
    assert cash == 5000.0


def test_signal_followup_current_status():
    # LONG entry 100, T1 110, stop 95 — price 112 → target reached
    s = sf._current_status("LONG", 100, 95, 110, 112)
    assert s["state"] == "TARGET1_REACHED" and s["unrealized_pct"] == 12.0
    s2 = sf._current_status("LONG", 100, 95, 110, 94)
    assert s2["state"] == "STOPPED"
