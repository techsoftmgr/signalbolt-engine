"""Unit test — chart_read.analyze() honors the timeframe param (1Day vs 1Hour).
Additive; get_bars is monkeypatched so it's offline + deterministic.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd


def _wire(monkeypatch):
    import engine.alpaca_client as ac
    calls = []

    def fake_get_bars(sym, timeframe, days=2):
        calls.append(timeframe)
        n = 250
        idx = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
        close = np.linspace(100, 120, n) + np.sin(np.arange(n)) * 1.5
        o = close - 0.3
        h = np.maximum(o, close) + 1
        l = np.minimum(o, close) - 1
        return pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                             "volume": [1_000_000.0] * n}, index=idx)

    monkeypatch.setattr(ac, "get_bars", fake_get_bars)
    return calls


def test_analyze_default_is_daily(monkeypatch):
    calls = _wire(monkeypatch)
    from engine import chart_read
    r = chart_read.analyze("TST")
    assert r is not None
    assert r["timeframe"] == "1Day"
    assert calls and calls[0] == "1Day"        # base fetch is daily


def test_analyze_hourly_timeframe(monkeypatch):
    calls = _wire(monkeypatch)
    from engine import chart_read
    r = chart_read.analyze("TST", timeframe="1Hour")
    assert r is not None
    assert r["timeframe"] == "1Hour"
    assert calls and calls[0] == "1Hour"        # base fetch honored the override


def test_analyze_unknown_tf_falls_back_to_daily(monkeypatch):
    calls = _wire(monkeypatch)
    from engine import chart_read
    r = chart_read.analyze("TST", timeframe="bogus")
    assert r["timeframe"] == "1Day"
    assert calls[0] == "1Day"
