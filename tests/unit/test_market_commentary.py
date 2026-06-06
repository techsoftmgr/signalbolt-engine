"""Unit tests — Market Tape (market_commentary) + econ_calendar. Additive.

Network deps (Alpaca / Finnhub / pulse) are monkeypatched so the tests are
deterministic and offline.
"""
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import market_commentary as mc
from engine import econ_calendar as ec


# ── policy/headline filter ────────────────────────────────────────────────────
def test_policy_headlines_filters_market_movers(monkeypatch):
    import engine.alpaca_client as ac
    monkeypatch.setattr(ac, "get_multi_news", lambda *a, **k: [
        {"headline": "Trump threatens new tariffs on imports", "summary": "...", "created_at": "2026-06-06T12:00:00Z", "url": "u1"},
        {"headline": "Acme Corp announces new logo", "summary": "...", "created_at": "2026-06-06T11:00:00Z"},
        {"headline": "Fed signals rate cut path", "summary": "...", "created_at": "2026-06-06T10:00:00Z"},
        {"headline": "Local bakery wins award", "summary": "...", "created_at": "2026-06-06T09:00:00Z"},
    ])
    out = mc._policy_headlines(limit=6)
    titles = [e["title"] for e in out]
    assert any("tariff" in t.lower() for t in titles)
    assert any("rate cut" in t.lower() for t in titles)
    assert not any("bakery" in t.lower() or "logo" in t.lower() for t in titles)
    assert all(e["type"] == "POLICY" for e in out)


def test_policy_headlines_empty_on_news_failure(monkeypatch):
    import engine.alpaca_client as ac
    monkeypatch.setattr(ac, "get_multi_news", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    assert mc._policy_headlines() == []


# ── phase detection ───────────────────────────────────────────────────────────
def test_phase_open_premarket_afterhours(monkeypatch):
    import engine.session_classifier as sc
    monkeypatch.setattr(sc, "is_market_open_today", lambda: True)
    monkeypatch.setattr(sc, "is_market_open_now", lambda: True)
    assert mc._phase(datetime(2026, 6, 5, 15, 0, tzinfo=timezone.utc)) == "open"
    monkeypatch.setattr(sc, "is_market_open_now", lambda: False)
    assert mc._phase(datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)) == "premarket"   # 12:00 UTC < 13:30
    assert mc._phase(datetime(2026, 6, 5, 21, 0, tzinfo=timezone.utc)) == "afterhours"  # 21:00 UTC
    monkeypatch.setattr(sc, "is_market_open_today", lambda: False)
    assert mc._phase(datetime(2026, 6, 6, 15, 0, tzinfo=timezone.utc)) == "closed"


# ── build assembly (internals stubbed) ────────────────────────────────────────
def test_build_assembles_and_never_raises(monkeypatch):
    monkeypatch.setattr(mc, "_phase", lambda now: "open")
    monkeypatch.setattr(mc, "_market_bias", lambda: {"bias": "risk-off", "vix": 22.5, "regime_type": "PANIC", "above_200ma": False})
    monkeypatch.setattr(mc, "_index_events", lambda phase: [
        {"time": "2026-06-06T14:00:00Z", "type": "VWAP", "tone": "bearish", "severity": 2, "title": "S&P 500: Lost VWAP", "detail": "..."},
    ])
    monkeypatch.setattr(mc, "_sector_event", lambda: [
        {"time": "2026-06-06T14:05:00Z", "type": "SECTOR", "tone": "bearish", "severity": 1, "title": "Sector leadership: XLU SELL", "detail": "..."},
    ])
    monkeypatch.setattr(mc, "_policy_headlines", lambda limit=6: [
        {"time": "2026-06-06T15:00:00Z", "type": "POLICY", "tone": "neutral", "severity": 2, "title": "Fed signals rate cut", "detail": "..."},
    ])
    monkeypatch.setattr(mc, "_gap_event", lambda now: [])
    import engine.econ_calendar as _ec
    monkeypatch.setattr(_ec, "today_and_upcoming", lambda now=None, days=7: {
        "today": [{"event": "FOMC decision day", "impact": "high"}], "upcoming": [], "has_feed": False})

    out = mc.build(datetime(2026, 6, 6, 15, 0, tzinfo=timezone.utc))
    assert out["available"] is True
    assert out["phase"] == "open"
    assert out["bias"] == "risk-off"
    assert "RISK-OFF" in out["summary"] and "VIX 22.5" in out["summary"]
    assert "FOMC" in out["summary"]
    # newest-first ordering by time
    times = [e["time"] for e in out["events"]]
    assert times == sorted(times, reverse=True)
    assert out["catalysts"] and out["catalysts"][0]["event"] == "FOMC decision day"
    assert "not financial advice" in out["disclaimer"].lower()
    assert len(out["events"]) <= mc._MAX_EVENTS


def test_build_degrades_when_everything_fails(monkeypatch):
    # all sub-blocks raise → build still returns a usable, neutral object
    for name in ("_index_events", "_sector_event", "_policy_headlines", "_gap_event"):
        monkeypatch.setattr(mc, name, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(mc, "_market_bias", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    out = mc.build()
    assert out["available"] in (True, False)   # never raises
    assert "disclaimer" in out


# ── econ calendar graceful fallback ───────────────────────────────────────────
def test_econ_calendar_no_key_degrades(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    out = ec.today_and_upcoming(datetime(2026, 6, 6, tzinfo=timezone.utc))
    assert out["has_feed"] is False
    assert isinstance(out["today"], list) and isinstance(out["upcoming"], list)


def test_econ_calendar_never_raises(monkeypatch):
    import engine.econ_calendar as _e
    monkeypatch.setattr(_e, "_finnhub_events", lambda days: (_ for _ in ()).throw(RuntimeError("boom")))
    out = _e.today_and_upcoming()
    assert "today" in out and "upcoming" in out
