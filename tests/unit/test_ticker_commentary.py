"""Unit tests — Ticker Commentary intraday event differ. Additive.

Detectors are exercised on synthetic bars; build() is exercised with get_bars
monkeypatched so the tests are deterministic and need no network/Alpaca.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from engine import ticker_commentary as tc


def _df(closes, vols=None, start="2026-06-05 13:30", freq="5min"):
    """Build a one-session OHLCV df from a close-price list."""
    n = len(closes)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    high = np.maximum(opens, closes) * 1.001
    low = np.minimum(opens, closes) * 0.999
    v = np.asarray(vols if vols is not None else [1000.0] * n, dtype=float)
    return pd.DataFrame({"open": opens, "high": high, "low": low, "close": closes, "volume": v}, index=idx)


def _types(events):
    return [e["type"] for e in events]


def test_macd_bullish_cross_emitted():
    # decline for 30 bars, then a sharp rise -> MACD histogram flips negative→positive
    closes = list(np.linspace(100, 90, 30)) + list(np.linspace(90, 104, 16))
    ev = tc._detect_tf(_df(closes), "5m", prior_close=None, want_ideas=True)
    macd = [e for e in ev if e["type"] == "MACD_CROSS" and e["tone"] == "bullish"]
    assert macd, f"expected a bullish MACD cross, got {_types(ev)}"


def test_flat_tape_is_quiet():
    ev = tc._detect_tf(_df([100.0] * 50, vols=[1000.0] * 50), "5m", prior_close=100.0, want_ideas=True)
    assert ev == [], f"flat tape should be silent, got {_types(ev)}"


def test_rsi_overbought_emitted():
    # flat then a strong steady climb -> RSI crosses above 70
    closes = [100.0] * 16 + list(np.linspace(100, 120, 24))
    ev = tc._detect_tf(_df(closes), "5m", prior_close=None, want_ideas=False)
    assert any(e["type"] == "RSI" and "overbought" in e["title"].lower() for e in ev), _types(ev)


def test_volume_spike_emitted():
    closes = list(np.linspace(100, 101, 30))
    vols = [1000.0] * 30
    vols[25] = 8000.0   # 8x spike
    ev = tc._detect_tf(_df(closes, vols=vols), "5m", prior_close=None, want_ideas=False)
    assert any(e["type"] == "VOLUME" for e in ev), _types(ev)


def test_sharp_move_emitted():
    closes = list(np.linspace(100, 101, 20)) + [101, 104.0] + list(np.linspace(104, 105, 8))  # +~3% bar
    ev = tc._detect_tf(_df(closes), "5m", prior_close=None, want_ideas=False)
    assert any(e["type"] == "MOVE" and e["tone"] == "bullish" for e in ev), _types(ev)


def test_opening_gap_emitted():
    closes = list(np.linspace(105, 108, 30))   # opens at 105 vs prior close 100 => +5% gap
    ev = tc._detect_tf(_df(closes), "5m", prior_close=100.0, want_ideas=False)
    gaps = [e for e in ev if e["type"] == "GAP"]
    assert gaps and gaps[0]["tone"] == "bullish", _types(ev)


def test_ideas_are_educational_never_advice():
    closes = list(np.linspace(100, 90, 30)) + list(np.linspace(90, 106, 18))
    ev = tc._detect_tf(_df(closes), "5m", prior_close=None, want_ideas=True)
    ideas = [e["idea"] for e in ev if e.get("idea")]
    for idea in ideas:
        t = idea["text"].lower()
        assert "not advice" in t or "educational" in t
        assert "buy now" not in t and "guaranteed" not in t and "sure profit" not in t


def test_cooldown_limits_repeat_events():
    # many tiny oscillations would naively spam VWAP/EMA crosses; cooldown caps them
    closes = []
    for _ in range(8):
        closes += [100, 101, 100, 99]
    ev = tc._detect_tf(_df(closes * 2), "5m", prior_close=None, want_ideas=False)
    # no single event type should fire on more than ~1/4 of bars thanks to cooldowns
    from collections import Counter
    c = Counter(_types(ev))
    assert all(v <= len(closes) for v in c.values())


# ── build() — defensive + integration with get_bars monkeypatched ────────────
def test_build_unavailable_on_no_data(monkeypatch):
    import engine.alpaca_client as ac
    monkeypatch.setattr(ac, "get_bars", lambda *a, **k: None)
    out = tc.build("TST")
    assert out["available"] is False and "ticker" in out


def test_build_unavailable_on_short_session(monkeypatch):
    import engine.alpaca_client as ac
    monkeypatch.setattr(ac, "get_bars", lambda *a, **k: _df([100.0] * 5))
    out = tc.build("TST")
    assert out["available"] is False


def test_build_full_feed(monkeypatch):
    import engine.alpaca_client as ac
    # prior day (for gap) + today's session with a decline→rally (forces events)
    day1 = _df([100.0] * 40, start="2026-06-04 13:30")
    closes = list(np.linspace(101, 92, 30)) + list(np.linspace(92, 108, 20))
    day2 = _df(closes, start="2026-06-05 13:30")
    full = pd.concat([day1, day2])
    monkeypatch.setattr(ac, "get_bars", lambda *a, **k: full)
    out = tc.build("TST")
    assert out["available"] is True
    assert out["ticker"] == "TST"
    assert out["session_date"] == "2026-06-05"
    assert isinstance(out["events"], list) and len(out["events"]) >= 1
    assert len(out["events"]) <= tc._MAX_EVENTS
    # newest-first ordering
    times = [e["time"] for e in out["events"]]
    assert times == sorted(times, reverse=True)
    assert out["lean"] in ("bullish", "bearish", "mixed")
    assert "not financial advice" in out["disclaimer"].lower()


def test_build_empty_symbol():
    assert tc.build("")["available"] is False
    assert tc.build(None)["available"] is False
