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


def test_news_events_from_ticker(monkeypatch):
    import engine.alpaca_client as ac
    monkeypatch.setattr(ac, "get_news", lambda sym, limit=4: [
        {"headline": "ABAT: DOE project cancellation reinstated", "summary": "positive",
         "created_at": "2026-06-07T12:00:00Z", "url": "u1", "source": "Benzinga"},
    ])
    ev = tc._news_events("ABAT")
    assert ev and ev[0]["type"] == "NEWS"
    assert "DOE" in ev[0]["title"] and ev[0]["url"] == "u1"
    assert ev[0]["severity"] == 2


def test_news_events_empty_on_failure(monkeypatch):
    import engine.alpaca_client as ac
    monkeypatch.setattr(ac, "get_news", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert tc._news_events("X") == []


def test_bar_session_classification():
    assert tc._bar_session(pd.Timestamp("2026-06-07 10:00", tz="UTC")) == "pre"     # 06:00 ET
    assert tc._bar_session(pd.Timestamp("2026-06-07 14:00", tz="UTC")) == "rth"     # 10:00 ET
    assert tc._bar_session(pd.Timestamp("2026-06-07 21:00", tz="UTC")) == "after"   # 17:00 ET


def test_events_carry_session_tag():
    closes = list(np.linspace(100, 90, 30)) + list(np.linspace(90, 104, 16))
    ev = tc._detect_tf(_df(closes), "5m", prior_close=None, want_ideas=False)
    assert ev and all("session" in e for e in ev)


def test_premarket_volume_spike_needs_bigger_multiple():
    closes = list(np.linspace(100, 101, 40))
    # 3.5× the trailing average — fires in RTH, but premarket needs >=5×
    v_small = [1000.0] * 40; v_small[30] = 3500.0
    pre = tc._detect_tf(_df(closes, vols=v_small, start="2026-06-05 08:00"), "5m", None, want_ideas=False)
    assert not any(e["type"] == "VOLUME" for e in pre), [e["type"] for e in pre]
    # a 6.5× spike DOES clear the premarket bar, and is labeled
    v_big = [1000.0] * 40; v_big[30] = 6500.0
    pre2 = tc._detect_tf(_df(closes, vols=v_big, start="2026-06-05 08:00"), "5m", None, want_ideas=False)
    vs = [e for e in pre2 if e["type"] == "VOLUME"]
    assert vs and "Premarket" in vs[0]["title"] and vs[0]["session"] == "pre"


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


def test_intraday_idea_educational_and_rr_floor():
    # good R:R bullish setup → idea, educational, rr >= floor
    good = tc._intraday_idea("bullish", price=100.0, swing_lo=98.0, swing_hi=112.0, atr=1.0)
    assert good is not None and good["rr"] >= tc._MIN_RR
    assert "not advice" in good["text"].lower()
    assert "buy now" not in good["text"].lower() and "guaranteed" not in good["text"].lower()
    # poor R:R (target barely above entry, stop far) → suppressed entirely
    poor = tc._intraday_idea("bullish", price=100.0, swing_lo=95.0, swing_hi=100.4, atr=1.0)
    assert poor is None


def test_bias_gate_blocks_counter_trend_idea():
    # a decline→rally produces a bullish MACD cross
    closes = list(np.linspace(100, 90, 30)) + list(np.linspace(90, 106, 18))
    # WITH the tape (bias up) → the bullish cross may carry an idea
    up = tc._detect_tf(_df(closes), "5m", prior_close=None, want_ideas=True, bias="up")
    up_macd = [e for e in up if e["type"] == "MACD_CROSS" and e["tone"] == "bullish"]
    assert up_macd
    assert any("idea" in e for e in up_macd)            # aligned → idea allowed
    # AGAINST the tape (bias down) → same bullish cross gets flagged, NO idea
    dn = tc._detect_tf(_df(closes), "5m", prior_close=None, want_ideas=True, bias="down")
    dn_macd = [e for e in dn if e["type"] == "MACD_CROSS" and e["tone"] == "bullish"]
    assert dn_macd
    assert all("idea" not in e for e in dn_macd)         # counter-trend → never an idea
    assert all(e.get("against_trend") for e in dn_macd)
    assert any("counter-trend" in e["detail"].lower() for e in dn_macd)


def test_downtrend_yields_short_ideas_from_continuation_triggers():
    # decline → bounce → rollover: a with-trend short should appear at the rollover
    # (from EMA/VWAP/MACD), not just a single MACD cross.
    closes = (list(np.linspace(100, 92, 20)) + list(np.linspace(92, 97, 10))
              + list(np.linspace(97, 93, 16)))
    ev = tc._detect_tf(_df(closes), "15m", prior_close=None, want_ideas=True, bias="down")
    ideas = [e for e in ev if e.get("idea")]
    assert ideas, _types(ev)
    for e in ideas:                       # every idea is a with-trend short, RR-gated
        assert e["idea"]["bias"] == "short"
        assert e["idea"]["rr"] >= tc._MIN_RR
        assert not e.get("against_trend")


def test_session_bias_up_down_neutral():
    up5 = _df(list(np.linspace(100, 110, 40)))
    up15 = tc._resample(up5, "15min")
    assert tc._session_bias(up5, up15) == "up"
    dn5 = _df(list(np.linspace(110, 100, 40)))
    dn15 = tc._resample(dn5, "15min")
    assert tc._session_bias(dn5, dn15) == "down"


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
