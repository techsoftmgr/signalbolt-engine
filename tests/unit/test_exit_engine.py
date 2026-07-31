"""
Exit engine — multi-factor confluence exit. Pins the two mechanisms:
  1. giveback cap (profit protection) fires on a round-trip of a real run;
  2. confluence exit requires a PRICE ANCHOR — momentum/flow alone can NEVER exit
     (the whipsaw the owner flagged: bearish CMF/RSI → we bailed → trend resumed).
"""
import numpy as np
import pandas as pd

from engine import exit_engine as ex


def _df(closes, vol=None, highs=None, lows=None):
    closes = np.asarray(closes, float)
    n = len(closes)
    return pd.DataFrame({
        "open": closes,
        "high": closes + 0.5 if highs is None else highs,
        "low":  closes - 0.5 if lows  is None else lows,
        "close": closes,
        "volume": [1e6] * n if vol is None else vol,
    })


# ── giveback cap ──────────────────────────────────────────────────────────────
def test_giveback_floor_arms_only_after_a_real_run():
    # peaked +3% (< 5% arm) → no floor yet
    assert ex.giveback_floor("LONG", 100, 103, ex.DEFAULT) is None
    # peaked +10% → floor locks half → +5% → 105
    assert abs(ex.giveback_floor("LONG", 100, 110, ex.DEFAULT) - 105.0) < 1e-6
    # SHORT symmetric: entry 100, peak 90 (+10% favorable) → floor at 95
    assert abs(ex.giveback_floor("SHORT", 100, 90, ex.DEFAULT) - 95.0) < 1e-6


def test_giveback_cap_exits_on_roundtrip():
    # AAPL-style: ran +10% (peak 110), now back to +4% (104) → below the +5% floor
    df = _df(np.linspace(100, 104, 40))
    r = ex.evaluate("LONG", entry=100, price=104, peak=110, daily_df=df)
    assert r["exit"] and r["reason"] == "giveback_cap"


def test_giveback_cap_holds_while_above_floor():
    df = _df(np.linspace(100, 108, 40))
    r = ex.evaluate("LONG", entry=100, price=108, peak=110, daily_df=df)  # +8% > +5% floor
    assert not r["exit"]


# ── confluence: PRICE ANCHOR REQUIRED ────────────────────────────────────────
def test_momentum_alone_never_exits_without_price_anchor():
    # Uptrend, then a shallow pullback that STAYS above the 20-EMA (no anchor) but
    # rolls momentum over on heavy volume. Must HOLD — this is the whipsaw guard.
    up = list(np.linspace(80, 120, 34))          # strong uptrend
    pull = [119, 118, 117]                        # mild dip, still well above EMA20
    closes = up + pull
    vol = [1e6] * 34 + [3e6, 3e6, 3e6]            # heavy down-volume
    # down bars close near their low → bearish CMF
    highs = np.array(closes) + 0.3
    lows = np.array(closes) - 0.3
    df = _df(closes, vol=vol, highs=highs, lows=lows)
    r = ex.evaluate("LONG", entry=80, price=117, peak=120, daily_df=df)
    assert r["anchor"] is False
    assert r["exit"] is False        # momentum/flow corroboration cannot exit alone


def test_confluence_exits_when_price_breaks_with_corroboration():
    # Uptrend then a DECISIVE breakdown: close drops below the 20-EMA (anchor) with
    # RSI rollover + MACD turning down → score clears threshold → exit.
    up = list(np.linspace(80, 120, 30))
    dn = [116, 110, 104, 98, 94]                  # sharp reversal through the MA
    closes = up + dn
    vol = [1e6] * 30 + [2e6] * 5
    highs = np.array(closes) + 0.3
    lows = np.array(closes) - 0.3
    df = _df(closes, vol=vol, highs=highs, lows=lows)
    # entry near the peak (small run) so the giveback cap doesn't arm — isolate the
    # confluence trend-exit path.
    r = ex.evaluate("LONG", entry=115, price=94, peak=120, daily_df=df)
    assert r["anchor"] is True
    assert r["exit"] is True and r["reason"] == "confluence_break"


def test_anchor_without_enough_corroboration_holds():
    # A single close just under the 20-EMA (anchor, +1.5) but nothing else → below
    # the 3.0 threshold → HOLD (don't bail on one soft MA poke).
    closes = list(np.linspace(100, 120, 34)) + [118.0]   # one bar dips slightly
    df = _df(closes)
    r = ex.evaluate("LONG", entry=100, price=118, peak=120, daily_df=df)
    # if this single bar isn't even below EMA20 it's trivially hold; if it is, score<thr
    assert r["exit"] is False


def test_short_mirror_confluence():
    # downtrend then a decisive bounce back ABOVE the 20-EMA → cover a short
    dn = list(np.linspace(120, 80, 30))
    up = [84, 90, 96, 102, 106]
    closes = dn + up
    df = _df(closes, vol=[1e6] * 30 + [2e6] * 5)
    # entry near the low (small favorable run) so the giveback cap doesn't arm.
    r = ex.evaluate("SHORT", entry=83, price=106, peak=80, daily_df=df)
    assert r["anchor"] is True and r["exit"] is True


def test_short_data_is_failsafe_hold():
    r = ex.evaluate("LONG", entry=100, price=101, peak=101, daily_df=_df([100, 101, 102]))
    assert r["action"] == "hold" and r["exit"] is False


# ── rollout gates: per-detector allowlist + kill switches ────────────────────
def test_manages_allowlist(monkeypatch):
    monkeypatch.delenv("SMART_EXIT_DETECTORS", raising=False)
    assert ex.manages("TREND_MOMENTUM") is True          # unset = all swings
    monkeypatch.setenv("SMART_EXIT_DETECTORS", "TREND_MOMENTUM, RS_PULLBACK")
    assert ex.manages("TREND_MOMENTUM") is True
    assert ex.manages("rs_pullback") is True             # case-insensitive
    assert ex.manages("PEAK_FORMING") is False           # winners left on old logic
    assert ex.manages(None) is False


def test_kill_switches_default_off(monkeypatch):
    for v in ("SMART_EXIT_ENABLED", "SMART_EXIT_SHADOW"):
        monkeypatch.delenv(v, raising=False)
    assert ex.enabled() is False and ex.shadow() is False
    monkeypatch.setenv("SMART_EXIT_SHADOW", "true")
    assert ex.shadow() is True
