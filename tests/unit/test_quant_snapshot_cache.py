"""Unit test — quant_score_service.snapshot per-ticker cache (the watchlist 'takes time' fix).

Custom (non-universe) tickers must be served from the per-ticker Redis cache when present, and a
freshly-scored one must be written back so the next load is warm (instead of re-fetching bars +
re-scoring every time)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import quant_score_service as qs
from engine import alpaca_client, regime_detector, ticker_fundamentals


class _FakeKV:
    def __init__(self): self.store = {}
    def get_json(self, k): return self.store.get(k)
    def set_json(self, k, v, ttl=None): self.store[k] = v
    def delete(self, k): self.store.pop(k, None)


def test_snapshot_uses_and_writes_per_ticker_cache(monkeypatch):
    fake = _FakeKV()
    monkeypatch.setattr(qs.cache, "kv", fake)
    fake.store[qs._SCORED_KEY] = []                       # universe scan empty → nothing warm there
    # AAA was scored on a PRIOR load → already in the per-ticker cache
    fake.store[qs._SNAP_KEY + "AAA"] = {"ticker": "AAA", "price": 10, "relativeVolume": 1.2, "rsi": 55}

    scored_calls = []
    def _fake_score(tk, *a, **k):
        scored_calls.append(tk)
        return {"ticker": tk, "price": 20, "relativeVolume": 2.0, "rsi": 60}
    monkeypatch.setattr(qs, "_score_ticker", _fake_score)
    monkeypatch.setattr(qs, "_get_long_bars", lambda *a, **k: {})
    monkeypatch.setattr(alpaca_client, "get_multi_bars", lambda *a, **k: {})
    monkeypatch.setattr(alpaca_client, "get_latest_prices", lambda *a, **k: {})
    monkeypatch.setattr(regime_detector, "detect", lambda *a, **k: {"regime_type": "X"})
    monkeypatch.setattr(ticker_fundamentals, "get", lambda *a, **k: {})

    out = qs.snapshot(["AAA", "BBB"])

    assert "AAA" not in scored_calls          # served from per-ticker cache → NOT re-scored
    assert scored_calls == ["BBB"]            # only the truly-missing one was scored
    assert qs._SNAP_KEY + "BBB" in fake.store # …and written back so next load is warm
    assert set(out.keys()) == {"AAA", "BBB"}  # both rows returned


def test_snapshot_prefers_universe_over_per_ticker_cache(monkeypatch):
    fake = _FakeKV()
    monkeypatch.setattr(qs.cache, "kv", fake)
    fake.store[qs._SCORED_KEY] = [{"ticker": "NVDA", "price": 100, "relativeVolume": 1.0, "rsi": 50}]
    monkeypatch.setattr(ticker_fundamentals, "get", lambda *a, **k: {})
    # _score_ticker must never run for a universe name
    monkeypatch.setattr(qs, "_score_ticker", lambda *a, **k: (_ for _ in ()).throw(AssertionError("scored a universe name")))
    monkeypatch.setattr(alpaca_client, "get_multi_bars", lambda *a, **k: {})
    monkeypatch.setattr(alpaca_client, "get_latest_prices", lambda *a, **k: {})

    out = qs.snapshot(["NVDA"])
    assert "NVDA" in out


def test_recent_max_rsi_latches_through_rollover():
    """Peak-detector latch: a name that WAS overbought must still register as recently-overbought
    after it starts rolling over (current RSI < 60) — the MSFT 466→370 miss."""
    rising = list(range(100, 145))                       # strictly up → RSI ~100
    assert qs._recent_max_rsi(rising) >= 70
    rolled = rising + [144, 142, 140, 138, 136]          # now rolling over (current RSI has dropped)
    assert qs._recent_max_rsi(rolled, lookback=10) >= 70 # latch still holds through the rollover
    flat = [100 + (i % 2) for i in range(45)]            # choppy/flat → never overbought
    assert qs._recent_max_rsi(flat) < 70
    assert qs._recent_max_rsi([1, 2, 3]) == 0.0          # too little history → safe 0
