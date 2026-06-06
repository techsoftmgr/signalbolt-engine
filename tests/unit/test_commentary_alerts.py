"""Unit tests — Commentary push alerts (V2). Additive.

Pure helpers (_alertworthy / _new_events / _format) + run() flow with cache.kv,
push.send_commentary_alert, ticker_commentary.build, and supabase all faked, so
the tests are deterministic and need no Redis / network / Alpaca.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import commentary_alerts as ca
from engine import cache as cache_mod
from engine import push as push_mod
from engine import ticker_commentary as tc_mod


def _ev(t, typ, tone="bullish", sev=2, **extra):
    return {"time": t, "type": typ, "tone": tone, "severity": sev,
            "title": f"{typ} ({extra.get('tf','5m')})", "detail": "detail here", **extra}


# ── pure helpers ──────────────────────────────────────────────────────────────
def test_alertworthy_filters_type_and_severity():
    assert ca._alertworthy(_ev("t", "MACD_CROSS", sev=3))
    assert ca._alertworthy(_ev("t", "MOVE", sev=2))
    assert not ca._alertworthy(_ev("t", "HOD", sev=1))      # noisy, in-feed only
    assert not ca._alertworthy(_ev("t", "RSI", sev=1))
    assert not ca._alertworthy(_ev("t", "VWAP", sev=1))     # below severity floor


def test_new_events_respects_watermark_and_sorts():
    events = [
        _ev("2026-06-05T15:00:00+00:00", "MACD_CROSS", sev=3),
        _ev("2026-06-05T13:50:00+00:00", "VWAP", sev=2),
        _ev("2026-06-05T14:30:00+00:00", "HOD", sev=1),      # not alert-worthy
        _ev("2026-06-05T14:40:00+00:00", "ORB", sev=2),
    ]
    fresh = ca._new_events(events, "2026-06-05T14:00:00+00:00")
    # only alert-worthy events strictly after the watermark, ascending
    assert [e["type"] for e in fresh] == ["ORB", "MACD_CROSS"]


def test_new_events_none_watermark_returns_all_alertworthy():
    events = [_ev("2026-06-05T13:50:00+00:00", "VWAP", sev=2),
              _ev("2026-06-05T14:00:00+00:00", "MACD_CROSS", sev=3)]
    fresh = ca._new_events(events, None)
    assert len(fresh) == 2


def test_format_strips_tf_and_appends_idea():
    ev = _ev("t", "MACD_CROSS", sev=3, tf="15m",
             idea={"bias": "long", "text": "Intraday: setup favors a long. Educational, not advice."})
    ev["title"] = "MACD bullish crossover (15m)"
    title, body = ca._format("HOOD", ev)
    assert "(15m)" not in title and "HOOD" in title and "MACD bullish crossover" in title
    assert "not advice" in body.lower()


# ── run() flow ──────────────────────────────────────────────────────────────
class _FakeKV:
    def __init__(self): self.d = {}
    def get_json(self, k): return self.d.get(k)
    def set_json(self, k, v, ttl): self.d[k] = v


class _Q:
    def __init__(self, data): self._data = data
    def select(self, *a, **k): return self
    def execute(self): return type("R", (), {"data": self._data})()


class _FakeSB:
    def __init__(self, rows): self.rows = rows
    def table(self, *_a, **_k): return _Q(self.rows)


def _wire(monkeypatch, feed_holder, sent):
    monkeypatch.setattr(cache_mod, "kv", _FakeKV())
    monkeypatch.setattr(tc_mod, "build", lambda sym: feed_holder["feed"])
    def _send(ticker, title, body, event_type=None, sb=None):
        sent.append({"ticker": ticker, "event": event_type, "title": title}); return 1
    monkeypatch.setattr(push_mod, "send_commentary_alert", _send)


def test_run_disabled_when_env_off(monkeypatch):
    monkeypatch.delenv("COMMENTARY_ALERTS_ENABLED", raising=False)
    sent = []
    _wire(monkeypatch, {"feed": {"available": True, "events": []}}, sent)
    out = ca.run(_FakeSB([{"ticker": "HOOD"}]))
    assert out.get("disabled") is True
    assert sent == []


def test_run_cold_start_seeds_no_push(monkeypatch):
    monkeypatch.setenv("COMMENTARY_ALERTS_ENABLED", "true")
    sent = []
    feed = {"available": True, "events": [_ev("2026-06-05T14:00:00+00:00", "MACD_CROSS", sev=3)]}
    _wire(monkeypatch, {"feed": feed}, sent)
    out = ca.run(_FakeSB([{"ticker": "HOOD"}]))
    assert out["seeded"] == 1 and out["alerts"] == 0
    assert sent == []   # first ever scan never pushes


def test_run_pushes_new_event_on_second_scan(monkeypatch):
    monkeypatch.setenv("COMMENTARY_ALERTS_ENABLED", "1")
    sent = []
    kv = _FakeKV()
    monkeypatch.setattr(cache_mod, "kv", kv)
    holder = {"feed": {"available": True, "events": [
        _ev("2026-06-05T14:00:00+00:00", "VWAP", sev=2),
    ]}}
    monkeypatch.setattr(tc_mod, "build", lambda sym: holder["feed"])
    monkeypatch.setattr(push_mod, "send_commentary_alert",
                        lambda *a, **k: (sent.append(a) or 1))
    sb = _FakeSB([{"ticker": "HOOD"}])

    ca.run(sb)                                   # scan 1: cold start → seed, no push
    assert sent == []
    holder["feed"] = {"available": True, "events": [
        _ev("2026-06-05T14:00:00+00:00", "VWAP", sev=2),
        _ev("2026-06-05T15:00:00+00:00", "MACD_CROSS", sev=3),   # NEW, alert-worthy
    ]}
    out = ca.run(sb)                             # scan 2: the new MACD cross pushes
    assert out["alerts"] == 1
    assert len(sent) == 1


def test_run_caps_per_ticker_per_day(monkeypatch):
    monkeypatch.setenv("COMMENTARY_ALERTS_ENABLED", "true")
    sent = []
    kv = _FakeKV()
    # pre-seed the watermark so events count as "new" (skip cold start)
    kv.set_json("cmt_seen:HOOD", {"t": "2026-06-05T13:00:00+00:00"}, 1)
    monkeypatch.setattr(cache_mod, "kv", kv)
    many = [_ev(f"2026-06-05T14:0{i}:00+00:00", "MACD_CROSS", sev=3) for i in range(6)]
    monkeypatch.setattr(tc_mod, "build", lambda sym: {"available": True, "events": many})
    monkeypatch.setattr(push_mod, "send_commentary_alert", lambda *a, **k: (sent.append(a) or 1))
    out = ca.run(_FakeSB([{"ticker": "HOOD"}]))
    assert out["alerts"] == ca._MAX_PER_TICKER_DAY
    assert len(sent) == ca._MAX_PER_TICKER_DAY
    assert out["skipped"] >= 1


def test_run_handles_unavailable_feed(monkeypatch):
    monkeypatch.setenv("COMMENTARY_ALERTS_ENABLED", "true")
    sent = []
    _wire(monkeypatch, {"feed": {"available": False, "note": "no data"}}, sent)
    out = ca.run(_FakeSB([{"ticker": "HOOD"}]))
    assert out["alerts"] == 0 and sent == []
