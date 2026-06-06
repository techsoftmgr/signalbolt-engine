"""Unit tests — Phase 2 Trader Home AI briefing (pure). Additive."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine.phase2 import trader_home as th


def test_briefing_assembles_from_modules():
    threat = {"level": "YELLOW", "threat_score": 40, "reasons": ["VIX elevated (22)"]}
    watch = [{"ticker": "NVDA", "priority": 80, "why": "earnings imminent"},
             {"ticker": "AAPL", "priority": 20, "why": "no notable change"}]
    comm = [{"ticker": "TSLA", "verdict": "REAL_MOMENTUM"}]
    txt = th.briefing(threat, watch, comm, "TRENDING_BULL", active_signals=3)
    assert "Good morning" in txt
    assert "Trending Bull" in txt
    assert "YELLOW" in txt and "VIX elevated" in txt
    assert "NVDA" in txt and "earnings imminent" in txt
    assert "TSLA" in txt                       # confirmed buzz surfaced
    assert "3 active signal" in txt
    assert "not financial advice" in txt.lower()


def test_briefing_handles_empty():
    txt = th.briefing(None, None, None, None, None)
    assert "Good morning" in txt and "educational" in txt.lower()


def test_briefing_no_urgent_watch():
    txt = th.briefing({"level": "GREEN", "threat_score": 10},
                      [{"ticker": "KO", "priority": 15, "why": "quiet"}], None, "LOW_VOL")
    assert "No watchlist names flag as urgent" in txt
