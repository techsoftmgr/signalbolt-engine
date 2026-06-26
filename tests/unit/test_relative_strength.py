"""
relative_strength.is_rs_leader — the RS exemption to the regime long-veto.

Encodes the study's `highRS + uptrend-intact` definition: outperforming SPY 20d
AND rising 20-EMA AND above 50-SMA. Must FAIL CLOSED (return False) on any data
gap so the protective regime veto stays in place.
"""
import numpy as np
import pandas as pd
import pytest

from engine import relative_strength as rs


def _df(closes):
    return pd.DataFrame({"close": np.array(closes, dtype=float),
                         "open": closes, "high": closes, "low": closes,
                         "volume": [1e6] * len(closes)})


def test_rs_leader_true_when_outperforms_and_uptrend(monkeypatch):
    # Steady uptrend over 60 bars: rising 20-EMA, above 50-SMA, +20d return strong.
    closes = list(np.linspace(60, 100, 60))
    monkeypatch.setattr(rs, "_spy_ret20", lambda: 0.02)   # SPY only +2% over 20d
    ok, detail = rs.is_rs_leader("HOOD", daily_df=_df(closes))
    assert ok is True
    assert detail["ema20_rising"] is True and detail["above_sma50"] is True
    assert detail["rs_vs_spy_pct"] > 0


def test_rs_laggard_false_when_underperforms_spy(monkeypatch):
    closes = list(np.linspace(98, 100, 60))           # barely up (+~2% over 20d)
    monkeypatch.setattr(rs, "_spy_ret20", lambda: 0.10)   # SPY +10% → we lag
    ok, detail = rs.is_rs_leader("LAG", daily_df=_df(closes))
    assert ok is False


def test_rs_downtrend_false(monkeypatch):
    closes = list(np.linspace(100, 60, 60))           # falling → ema not rising, below 50-SMA
    monkeypatch.setattr(rs, "_spy_ret20", lambda: -0.30)  # even vs a worse SPY
    ok, detail = rs.is_rs_leader("DN", daily_df=_df(closes))
    assert ok is False


def test_fails_closed_on_insufficient_bars(monkeypatch):
    monkeypatch.setattr(rs, "_spy_ret20", lambda: 0.0)
    ok, detail = rs.is_rs_leader("X", daily_df=_df(list(range(10))))
    assert ok is False and "insufficient" in detail.get("reason", "")


def test_fails_closed_when_spy_unavailable(monkeypatch):
    closes = list(np.linspace(60, 100, 60))
    monkeypatch.setattr(rs, "_spy_ret20", lambda: None)   # benchmark down
    ok, _ = rs.is_rs_leader("HOOD", daily_df=_df(closes))
    assert ok is False


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("RS_EXEMPTION_ENABLED", "false")
    assert rs.enabled() is False
    monkeypatch.setenv("RS_EXEMPTION_ENABLED", "true")
    assert rs.enabled() is True
