"""Unit tests — universal fire-time enrichment (signal_telemetry). Offline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import signal_telemetry as st


def test_enrich_fill_missing_no_clobber(monkeypatch):
    monkeypatch.setattr(st, "market_context", lambda tk: {"relativeVolume": 1.8, "ma20": 100.0, "rsi": 55})
    monkeypatch.setattr(st, "classify_asset", lambda tk: {"asset_class": "equity", "is_etf": False})
    monkeypatch.setattr(st, "capture", lambda sb, tk, d, s: ("TRENDING_BULL", {"sector": "tech"}))
    # ma20 already set by the detector -> must NOT be clobbered
    row = {"ticker": "AAPL", "direction": "LONG", "strategy_type": "day_trade",
           "score_breakdown": {"ma20": 99.0, "detector_source": "SMC"}}
    st.enrich_score_breakdown(None, row)
    sbd = row["score_breakdown"]
    assert sbd["ma20"] == 99.0            # preserved (fill-missing only)
    assert sbd["relativeVolume"] == 1.8   # added
    assert sbd["rsi"] == 55
    assert sbd["asset_class"] == "equity"
    assert sbd["study"] == {"sector": "tech"}
    assert row["regime_type"] == "TRENDING_BULL"


def test_enrich_skips_capture_when_study_present(monkeypatch):
    called = {"n": 0}
    def fake_capture(*a, **k):
        called["n"] += 1
        return ("X", {})
    monkeypatch.setattr(st, "market_context", lambda tk: {})
    monkeypatch.setattr(st, "classify_asset", lambda tk: {})
    monkeypatch.setattr(st, "capture", fake_capture)
    # detector (e.g. breakdown) already embedded its own study blob + regime
    row = {"ticker": "T", "direction": "SHORT", "strategy_type": "breakdown",
           "regime_type": "BEAR", "score_breakdown": {"study": {"sector": "x"}}}
    st.enrich_score_breakdown(None, row)
    assert called["n"] == 0               # capture() not re-run (idempotent)
    assert row["regime_type"] == "BEAR"   # preserved


def test_enrich_no_score_breakdown_is_noop():
    row = {"ticker": "T"}
    st.enrich_score_breakdown(None, row)   # must not raise
    assert "score_breakdown" not in row


def test_market_context_whitelists_and_misses(monkeypatch):
    from engine import cache
    monkeypatch.setattr(cache.kv, "get_json", lambda k: [
        {"ticker": "NVDA", "relativeVolume": 2.1, "ma20": 120.0, "rsi": 60, "junk": 1},
        {"ticker": "AAPL", "relativeVolume": 0.9},
    ])
    ctx = st.market_context("nvda")
    assert ctx["relativeVolume"] == 2.1 and ctx["ma20"] == 120.0 and ctx["rsi"] == 60
    assert "junk" not in ctx               # only whitelisted fields carried
    assert st.market_context("MISSING") == {}
