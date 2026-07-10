"""
quant_score_service full-row snap cache for non-universe tickers (the /overview
timeout fix). A buzz-alerted / custom ticker (GOOG — the scan uses GOOGL) must be
served from cache after the first cold compute instead of re-running every tap.
"""
from engine import quant_score_service as q
from engine import cache


def test_full_single_snap_roundtrip(monkeypatch):
    store: dict = {}
    monkeypatch.setattr(cache.kv, "get_json", lambda k: store.get(k))
    monkeypatch.setattr(cache.kv, "set_json", lambda k, v, ttl=None: store.__setitem__(k, v))

    assert q.cached_full_single("GOOG") == (None, None)          # cold
    q.store_full_single("GOOG", {"ticker": "GOOG", "rsi": 55}, "2026-07-09T00:00:00Z")
    row, as_of = q.cached_full_single("GOOG")
    assert row["rsi"] == 55 and as_of.startswith("2026")
    assert store  # something was written under the full-snap key


def test_full_single_snap_safe_on_blank_and_errors(monkeypatch):
    assert q.cached_full_single("") == (None, None)
    q.store_full_single("", {"x": 1}, "t")          # no-op, must not raise
    # a cache backend error fails closed, never raises
    monkeypatch.setattr(cache.kv, "get_json", lambda k: (_ for _ in ()).throw(RuntimeError("redis down")))
    assert q.cached_full_single("GOOG") == (None, None)
