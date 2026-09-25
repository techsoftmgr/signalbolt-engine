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


def test_replay_captures_lifecycle_past_close():
    # This is the "actual trade already closed, but smart-exit would've held/given back"
    # measurement: replay from entry over daily bars that extend past the real close.
    hist = list(np.linspace(80, 100, 40))         # pre-entry history (indicators)
    fwd = [100, 105, 110, 108, 102, 98, 95]        # ran +10% then round-tripped
    closes = hist + fwd
    idx = pd.date_range("2026-05-01", periods=len(closes), freq="D")
    df = pd.DataFrame({"open": closes, "high": np.array(closes) + 1, "low": np.array(closes) - 1,
                       "close": closes, "volume": [1e6] * len(closes)}, index=idx)
    r = ex.replay("LONG", entry=100, stop=90, daily_df=df, entry_date=idx[40].date())
    assert r is not None and r["exit_reason"] == "giveback_cap"
    assert 4.0 <= r["pnl_pct"] <= 6.5             # locked ~half of the ~+11% peak
    # insufficient data → None (fail-safe)
    assert ex.replay("LONG", 100, 90, _df([100, 101, 102])) is None


def test_kill_switches_default_off(monkeypatch):
    for v in ("SMART_EXIT_ENABLED", "SMART_EXIT_SHADOW"):
        monkeypatch.delenv(v, raising=False)
    assert ex.enabled() is False and ex.shadow() is False
    monkeypatch.setenv("SMART_EXIT_SHADOW", "true")
    assert ex.shadow() is True


# ── exhaustion / extension take-profit (observe-only, shadow) ────────────────
def test_extension_atr_flags_a_parabolic_stretch():
    # a flat base then a sharp spike → the last close sits far above the 20-EMA in ATR
    closes = list(np.linspace(100, 102, 34)) + [104, 110, 120]
    df = _df(closes)
    ext = ex.extension_atr("LONG", df)
    assert ext is not None and ext > 3           # clearly stretched in the trade's favour
    # a quiet, mean-hugging series is NOT extended
    assert ex.extension_atr("LONG", _df(list(np.linspace(100, 101, 40)))) < 2


def _daily(closes, highs=None):
    closes = np.asarray(closes, float)
    idx = pd.date_range("2026-05-01", periods=len(closes), freq="D")
    return pd.DataFrame({"open": closes, "high": closes + 1 if highs is None else np.asarray(highs, float),
                         "low": closes - 1, "close": closes, "volume": [1e6] * len(closes)}, index=idx)


def test_replay_ext_tp_books_into_an_exhaustion_spike():
    # entry NEAR the mean (realistic momentum entry), a modest run, then a 1-bar blow-off
    # spike, then a gap-down straight through the stop. The trail exits LOW (stop); the
    # exhaustion TP already banked a partial into the spike → blended pnl beats the trail.
    # (the ARM/PANW pattern from the 2026-09 ground-truth.)
    hist = list(np.linspace(99, 100, 40))
    closes = hist + [101, 102, 103, 125, 90]
    highs = [c + 1 for c in hist] + [102, 103, 104, 128, 96]
    lows = [c - 1 for c in hist] + [100, 101, 102, 120, 88]     # last bar gaps down through 92
    idx = pd.date_range("2026-05-01", periods=len(closes), freq="D")
    df = pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes,
                       "volume": [1e6] * len(closes)}, index=idx)
    ed = df.index[40].date()
    trail = ex.replay("LONG", 100, 92, df, entry_date=ed)
    hyb = ex.replay_ext_tp("LONG", 100, 92, df, entry_date=ed, k=4.0, scale=0.5)
    assert hyb is not None and hyb["ext_hit"] is True
    assert hyb["peak_ext"] is not None and hyb["peak_ext"] >= 4.0
    assert hyb["ext_pnl"] > hyb["trail_pnl"]          # the exhaustion leg banked the spike
    assert hyb["pnl_pct"] > trail["pnl_pct"]          # blended capture beats the pure trail


def test_replay_ext_tp_falls_back_to_trail_without_a_spike():
    # a steady grind that never reaches k×ATR → no exhaustion fill → pnl == the trail
    closes = list(np.linspace(80, 100, 40)) + list(np.linspace(100, 112, 8))
    df = _daily(closes)
    ed = df.index[40].date()
    trail = ex.replay("LONG", 100, 90, df, entry_date=ed)
    hyb = ex.replay_ext_tp("LONG", 100, 90, df, entry_date=ed, k=6.0, scale=0.5)
    assert hyb["ext_hit"] is False
    assert abs(hyb["pnl_pct"] - trail["pnl_pct"]) < 1e-6


def test_replay_ext_tp_never_books_a_loss_on_an_adverse_bounce():
    # entry then an immediate crash, then a dead-cat bounce that is "extended" above the
    # FALLING mean but still far BELOW entry. The exhaustion TP must NOT fire (a profit
    # tool never books a loss) — this is the CRWD -74% bug guard.
    hist = list(np.linspace(99, 100, 40))
    closes = hist + [92, 80, 70, 78, 74]        # crash then a bounce to 78 (below entry 100)
    highs = [c + 1 for c in hist] + [93, 81, 71, 82, 75]
    lows = [c - 1 for c in hist] + [88, 78, 68, 76, 72]
    idx = pd.date_range("2026-05-01", periods=len(closes), freq="D")
    df = pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes,
                       "volume": [1e6] * len(closes)}, index=idx)
    ed = df.index[40].date()
    hyb = ex.replay_ext_tp("LONG", 100, 60, df, entry_date=ed, k=2.0, scale=0.5)
    assert hyb["ext_hit"] is False                 # never armed / never a profitable fill
    assert hyb["pnl_pct"] == hyb["trail_pnl"]      # pure trail, no phantom -74% "profit"
