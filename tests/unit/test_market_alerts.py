"""Unit tests — Market Tape push alerts (market_alerts). Additive, offline."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import market_alerts as ma


class _FakeKV:
    def __init__(self):
        self.d = {}
    def get_json(self, k):
        return self.d.get(k)
    def set_json(self, k, v, ttl):
        self.d[k] = v


def _wire(monkeypatch, posts=None, events=None, market_open=True):
    import engine.cache as cache_mod
    import engine.push as push_mod
    import engine.social_feed as sf
    import engine.session_classifier as sc
    import engine.market_commentary as mc
    fake = _FakeKV()
    monkeypatch.setattr(cache_mod, "kv", fake)
    sent = {"social": [], "market": []}
    monkeypatch.setattr(push_mod, "send_social_alert", lambda a, t, u=None: (sent["social"].append((a, t)) or 1))
    monkeypatch.setattr(push_mod, "send_market_alert", lambda title, body, kind="market": (sent["market"].append(title) or 1))
    monkeypatch.setattr(sf, "recent_posts", lambda limit=15: posts or [])
    monkeypatch.setattr(sc, "is_market_open_now", lambda: market_open)
    monkeypatch.setattr(mc, "build", lambda: {"events": events or []})
    monkeypatch.setenv("MARKET_ALERTS_ENABLED", "true")
    return fake, sent


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MARKET_ALERTS_ENABLED", raising=False)
    assert ma.run() == {"disabled": True}


def test_social_cold_start_seeds_no_push(monkeypatch):
    posts = [{"author": "@realDonaldTrump", "text": "Tariffs!", "url": "u1"}]
    fake, sent = _wire(monkeypatch, posts=posts, market_open=False)
    out = ma.run()
    assert sent["social"] == []          # cold start pushes nothing
    assert out["seeded"] >= 1
    assert fake.get_json("mkt_social_init") is not None


def test_social_pushes_new_post_after_seed(monkeypatch):
    posts = [{"author": "@realDonaldTrump", "text": "Tariffs!", "url": "u1"}]
    fake, sent = _wire(monkeypatch, posts=posts, market_open=False)
    ma.run()                              # seed
    # a NEW post arrives
    posts.append({"author": "@realDonaldTrump", "text": "Fed should cut!", "url": "u2"})
    ma.run()
    assert any("Fed should cut" in t for _, t in sent["social"])
    # same post again → no duplicate push
    sent["social"].clear()
    ma.run()
    assert sent["social"] == []


def test_market_event_rth_gated(monkeypatch):
    ev = [{"type": "GAP", "severity": 2, "time": "2026-06-06T14:00:00Z", "title": "S&P 500: down 1.2%", "detail": "..."}]
    # market closed → no market push (even after seed)
    fake, sent = _wire(monkeypatch, events=ev, market_open=False)
    ma.run(); ma.run()
    assert sent["market"] == []


def test_market_event_pushes_new_after_seed(monkeypatch):
    ev = [{"type": "GAP", "severity": 2, "time": "2026-06-06T14:00:00Z", "title": "S&P 500: down 1.2%", "detail": "big gap"}]
    fake, sent = _wire(monkeypatch, events=ev, market_open=True)
    ma.run()                              # seeds watermark, no push
    assert sent["market"] == []
    ev.append({"type": "VWAP", "severity": 2, "time": "2026-06-06T15:00:00Z", "title": "S&P 500: lost VWAP", "detail": "..."})
    ma.run()
    assert any("lost VWAP" in t for t in sent["market"])
