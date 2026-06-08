"""Unit tests — per-ticker news push alerts (news_alerts). Additive, offline."""
import sys
import os
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import news_alerts as na


class _FakeKV:
    def __init__(self):
        self.d = {}
    def get_json(self, k):
        return self.d.get(k)
    def set_json(self, k, v, ttl):
        self.d[k] = v


class _SB:
    def __init__(self, tickers):
        self._t = tickers
    def table(self, *_a, **_k):
        return self
    def select(self, *_a, **_k):
        return self
    def execute(self):
        return type("R", (), {"data": [{"ticker": t} for t in self._t]})()


def _wire(monkeypatch, items, tickers=("ABAT",)):
    import engine.cache as cache_mod
    import engine.push as push_mod
    import engine.alpaca_client as ac
    fake = _FakeKV()
    monkeypatch.setattr(cache_mod, "kv", fake)
    sent = []
    monkeypatch.setattr(push_mod, "send_news_alert",
                        lambda tk, head, url=None, sb=None: (sent.append((tk, head)) or 1))
    monkeypatch.setattr(ac, "get_multi_news", lambda tks, limit=50: items)
    monkeypatch.setenv("NEWS_ALERTS_ENABLED", "true")
    return fake, sent, _SB(list(tickers))


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NEWS_ALERTS_ENABLED", raising=False)
    assert na.run(object()) == {"disabled": True}


def test_cold_start_seeds_then_pushes_new(monkeypatch):
    now = datetime(2026, 6, 7, 15, 0, tzinfo=timezone.utc)
    fresh = (now - timedelta(hours=1)).isoformat()
    items = [{"symbols": ["ABAT"], "headline": "DOE project reinstated", "url": "u1", "created_at": fresh}]
    fake, sent, sb = _wire(monkeypatch, items)
    na.run(sb, now=now)                       # cold start → seed, no push
    assert sent == []
    items.append({"symbols": ["ABAT"], "headline": "ABAT wins contract", "url": "u2", "created_at": fresh})
    na.run(sb, now=now)
    assert any("contract" in h for _, h in sent)
    sent.clear()
    na.run(sb, now=now)                       # same headline again → deduped
    assert sent == []


def test_stale_headline_not_pushed(monkeypatch):
    now = datetime(2026, 6, 7, 15, 0, tzinfo=timezone.utc)
    stale = (now - timedelta(hours=30)).isoformat()
    items = [{"symbols": ["ABAT"], "headline": "old news", "url": "u1", "created_at": stale}]
    fake, sent, sb = _wire(monkeypatch, items)
    na.run(sb, now=now)                       # cold seed
    items.append({"symbols": ["ABAT"], "headline": "another stale one", "url": "u2", "created_at": stale})
    na.run(sb, now=now)
    assert sent == []                         # unseen but stale → not pushed


def test_only_watched_symbols(monkeypatch):
    now = datetime(2026, 6, 7, 15, 0, tzinfo=timezone.utc)
    fresh = (now - timedelta(hours=1)).isoformat()
    # headline tagged for TSLA only; user watches ABAT → ignored
    items = [{"symbols": ["TSLA"], "headline": "TSLA thing", "url": "u1", "created_at": fresh}]
    fake, sent, sb = _wire(monkeypatch, items, tickers=("ABAT",))
    na.run(sb, now=now); na.run(sb, now=now)
    assert sent == []
