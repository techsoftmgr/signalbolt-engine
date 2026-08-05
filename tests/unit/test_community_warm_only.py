"""Community request path must NEVER cold-compute (the 'engine timeout' fix). With a
COLD cache, build_on_miss=False returns a fast 'warming' shape without touching the
heavy build (~9s trending / ~20s track_record). do-not-regress."""
from engine import social_insights as si


class _FakeKV:
    def __init__(self): self.store = {}
    def get_json(self, k): return self.store.get(k)
    def set_json(self, k, v, ttl=None): self.store[k] = v


def test_track_record_warm_only_does_not_build(monkeypatch):
    monkeypatch.setattr(si.cache, "kv", _FakeKV())          # cold cache

    class _BoomSB:  # any DB access ⇒ it tried to cold-compute on the request path
        def table(self, *a, **k): raise AssertionError("track_record cold-computed on request path")

    r = si.track_record(_BoomSB(), build_on_miss=False)
    assert r.get("warming") is True and r.get("ready") is False and r.get("buckets") == []


def test_enriched_trending_warm_only_does_not_build(monkeypatch):
    monkeypatch.setattr(si.cache, "kv", _FakeKV())          # cold cache
    from engine import social_sentiment
    monkeypatch.setattr(social_sentiment, "get_trending",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("trending cold-computed")))

    r = si.get_enriched_trending(object(), build_on_miss=False)
    assert r.get("warming") is True and r.get("trending") == []


def test_pulse_names_tickers_not_counts(monkeypatch):
    """The digest must NAME the tickers per verdict, not emit a bare count."""
    monkeypatch.setattr(si.cache, "kv", _FakeKV())
    rows = {"trending": [
        {"ticker": "SPCX", "verdict": {"key": "REAL_MOMENTUM"}},
        {"ticker": "AMD",  "verdict": {"key": "REAL_MOMENTUM"}},
        {"ticker": "HYPX", "verdict": {"key": "HYPE_FADING"}},
        {"ticker": "PMPX", "verdict": {"key": "PUMP_RISK"}},
    ]}
    monkeypatch.setattr(si, "get_enriched_trending", lambda *a, **k: rows)
    monkeypatch.setattr(si, "whats_changed", lambda *a, **k: {"newToday": []})

    text = " ".join(si.community_pulse(object())["bullets"])
    assert "SPCX" in text and "AMD" in text and "HYPX" in text and "PMPX" in text
    assert "name(s)" not in text          # no bare-count phrasing


def test_build_on_miss_default_true_preserves_worker_path(monkeypatch):
    # The worker (force=True) and any build_on_miss=True caller still build — the fix
    # only changes the request path.
    fake = _FakeKV()
    monkeypatch.setattr(si.cache, "kv", fake)
    fake.store[si._TRACK_CACHE_KEY] = {"ready": True, "buckets": [1, 2], "cached": True}
    assert si.track_record(object(), build_on_miss=False)["cached"] is True   # warm hit still served
