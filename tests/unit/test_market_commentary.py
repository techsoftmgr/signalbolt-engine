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


def test_build_caches_market_wide(monkeypatch):
    class _KV:
        def __init__(self): self.d = {}
        def get_json(self, k): return self.d.get(k)
        def set_json(self, k, v, ttl): self.d[k] = v
    import engine.cache as cache_mod
    monkeypatch.setattr(cache_mod, "kv", _KV())
    calls = {"n": 0}
    def _bias():
        calls["n"] += 1
        return {"bias": "neutral", "vix": 18, "regime_type": None, "above_200ma": True}
    monkeypatch.setattr(mc, "_phase", lambda now: "open")
    monkeypatch.setattr(mc, "_market_bias", _bias)
    monkeypatch.setattr(mc, "_internals", lambda: None)
    for name in ("_index_events", "_sector_event", "_social_events", "_policy_headlines", "_gap_event"):
        monkeypatch.setattr(mc, name, lambda *a, **k: [])
    import engine.econ_calendar as _ec
    monkeypatch.setattr(_ec, "today_and_upcoming", lambda now=None, days=7: {"today": [], "upcoming": [], "has_feed": False})
    mc.build()   # computes + caches
    mc.build()   # served from cache (no recompute)
    assert calls["n"] == 1


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


# ── V3: bias track record scoring ─────────────────────────────────────────────
def test_bias_correct_scoring():
    assert mc._bias_correct("risk-on", 1.2) is True       # bias up, SPY up → right
    assert mc._bias_correct("risk-on", -0.5) is False     # bias up, SPY down → wrong
    assert mc._bias_correct("risk-off", -1.0) is True     # bias down, SPY down → right
    assert mc._bias_correct("risk-off", 0.8) is False
    assert mc._bias_correct("neutral", 2.0) is None       # neutral isn't scored
    assert mc._bias_correct("risk-on", 0.1) is False      # inside the ±0.2% deadband


def test_log_and_score_bias_no_sb():
    assert mc.log_bias_snapshot(None)["logged"] == 0
    assert mc.score_bias_snapshots(None)["scored"] == 0


# ── SPY vs QQQ internals (leadership / divergence) ────────────────────────────
def test_internals_states(monkeypatch):
    def setp(spy, qqq):
        monkeypatch.setattr(mc, "_index_day_pct", lambda: {"SPY": spy, "QQQ": qqq})
        return mc._internals()
    assert setp(0.4, 1.3)["state"] == "growth_leading"      # Nasdaq leading
    assert setp(0.9, 0.1)["state"] == "broad_leading"       # S&P leading
    assert setp(0.6, -0.6)["divergent"] is True             # split → divergent
    assert setp(0.5, 0.6)["state"] == "in_line"             # together
    monkeypatch.setattr(mc, "_index_day_pct", lambda: {"SPY": 0.4})  # missing QQQ
    assert mc._internals() is None


def test_divergence_downgrades_bias_and_emits_event(monkeypatch):
    monkeypatch.setattr(mc, "_phase", lambda now: "open")
    monkeypatch.setattr(mc, "_market_bias", lambda: {"bias": "risk-on", "vix": 18, "regime_type": None, "above_200ma": True})
    monkeypatch.setattr(mc, "_internals", lambda: {"spy_pct": 0.5, "qqq_pct": -0.6, "spread": -1.1, "state": "divergent", "leader": "broad", "divergent": True})
    for name in ("_index_events", "_sector_event", "_social_events", "_policy_headlines", "_gap_event"):
        monkeypatch.setattr(mc, name, lambda *a, **k: [])
    import engine.econ_calendar as _ec
    monkeypatch.setattr(_ec, "today_and_upcoming", lambda now=None, days=7: {"today": [], "upcoming": [], "has_feed": False})
    out = mc.build(datetime(2026, 6, 6, 15, 0, tzinfo=timezone.utc))
    assert out["bias"] == "neutral"                          # split downgrades risk-on
    assert out["internals"]["divergent"] is True
    assert any(e["type"] == "DIVERGENCE" for e in out["events"])
    assert "diverging" in out["summary"]
