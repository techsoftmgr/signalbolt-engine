"""Unit tests for Market Pulse — pillar calculators + regime resolver (pure)."""
import numpy as np
import pandas as pd
import pytest

from engine.market_pulse import config as C
from engine.market_pulse import guidance, pillars, regime


def _df(closes, vols=None, highs=None, lows=None):
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": closes,
        "high": highs if highs is not None else closes,
        "low": lows if lows is not None else closes,
        "close": closes,
        "volume": vols if vols is not None else [1_000_000] * n,
    }, index=idx)


# ── Pillar 1: distribution days + the 5%-rise expiration rule ───────────────
def test_distribution_day_basic_count():
    # day 1 closes -0.5% on higher volume vs day 0 → 1 distribution day
    df = _df([100.0, 99.5], vols=[1_000_000, 1_500_000])
    assert pillars.distribution_days(df) == 1


def test_distribution_day_needs_higher_volume():
    df = _df([100.0, 99.5], vols=[1_000_000, 800_000])   # down but LOWER volume
    assert pillars.distribution_days(df) == 0


def test_distribution_day_5pct_rise_expiration():
    # DD on day 1 (down 0.5%, higher vol), then a later close 6% above its close
    # → that distribution day EXPIRES and must not count.
    closes = [100.0, 99.5, 100.0, 106.0]          # 106 >= 99.5*1.05 (=104.475)
    vols = [1_000_000, 1_500_000, 1_400_000, 1_400_000]
    assert pillars.distribution_days(_df(closes, vols)) == 0
    # Without the 5% rise it still counts:
    closes2 = [100.0, 99.5, 100.0, 101.0]
    assert pillars.distribution_days(_df(closes2, vols)) == 1


# ── Pillar 1b: stalling days ────────────────────────────────────────────────
def test_stalling_day_vs_healthy_up_day():
    # day1 = stalling: tiny gain (+0.1%), HIGHER volume, weak close (lower half)
    df = _df([100.0, 100.10], vols=[1_000_000, 1_500_000], highs=[100.0, 101.5], lows=[100.0, 99.0])
    assert pillars.stalling_days(df) == 1
    # healthy up day: big gain (+1.5%) → NOT stalling
    df2 = _df([100.0, 101.5], vols=[1_000_000, 1_500_000], highs=[100.0, 101.6], lows=[100.0, 100.0])
    assert pillars.stalling_days(df2) == 0
    # tiny gain but CLOSED STRONG (upper half of range) → NOT stalling
    df3 = _df([100.0, 100.10], vols=[1_000_000, 1_500_000], highs=[100.2, 100.2], lows=[99.0, 99.0])
    assert pillars.stalling_days(df3) == 0
    # tiny gain, weak close, but LOWER volume → NOT stalling
    df4 = _df([100.0, 100.10], vols=[1_000_000, 800_000], highs=[100.0, 101.5], lows=[100.0, 99.0])
    assert pillars.stalling_days(df4) == 0


def test_effective_dd_combination():
    from engine.market_pulse import job as mp_job
    assert mp_job._effective_dd_max(4, 0, 2, 0) == 5    # 4 + 0.5*2 = 5 → pressure
    assert mp_job._effective_dd_max(5, 0, 2, 0) == 6    # 5 + 1 = 6 → correction level
    assert mp_job._effective_dd_max(3, 0, 0, 0) == 3    # no stalling → unchanged (== dd_max)
    assert mp_job._effective_dd_max(2, 4, 0, 2) == 5    # max across SPY/QQQ: 4 + 0.5*2


def test_breadth_thrust():
    weak = [(80, 420)] * 8          # ratio ~0.16 (oversold)
    strong = [(440, 60)] * 8        # ratio ~0.88 (broad strength)
    # oversold → strong = bullish thrust (up); not a breakdown.
    ratio, up, down = pillars.breadth_thrust(weak + strong)
    assert up is True and down is False and ratio > C.BREADTH_THRUST_HIGH
    # MIRROR: strong → weak = bearish breakdown (down); not a thrust.
    ratio_b, up_b, down_b = pillars.breadth_thrust(strong + weak)
    assert down_b is True and up_b is False and ratio_b < C.BREADTH_THRUST_LOW
    # Flat ~0.5 → neither.
    ratio2, up2, down2 = pillars.breadth_thrust([(250, 250)] * 20)
    assert up2 is False and down2 is False
    # Too little data → (None, False, False).
    assert pillars.breadth_thrust([(250, 250)] * 3) == (None, False, False)


# ── Pillar 2: new highs / lows ──────────────────────────────────────────────
def test_net_new_highs_lows():
    n = C.HL_LOOKBACK + 5
    rising = np.linspace(50, 150, n)              # ends at a new high
    falling = np.linspace(150, 50, n)             # ends at a new low
    flat = np.full(n, 100.0)                      # neither
    bars = {"UP": _df(rising), "DOWN": _df(falling), "FLAT": _df(flat)}
    nh, nl, net = pillars.net_new_highs_lows(bars)
    assert nh == 1 and nl == 1 and net == 0


def test_short_history_skipped_for_new_highs():
    bars = {"SHORT": _df(np.linspace(50, 150, 100))}   # < 252 bars
    assert pillars.net_new_highs_lows(bars) == (0, 0, 0)


# ── Pillar 3: % above MAs ───────────────────────────────────────────────────
def test_pct_above_mas():
    n = C.SMA_SLOW + 10
    up = np.linspace(50, 150, n)                  # last close well above both MAs
    down = np.linspace(150, 50, n)                # last close below both MAs
    pct50, pct200 = pillars.pct_above_mas({"UP": _df(up), "DOWN": _df(down)})
    assert pct50 == 50.0 and pct200 == 50.0       # 1 of 2 above each


# ── Pillar 4: advance/decline + divergence ──────────────────────────────────
def test_advance_decline():
    bars = {
        "A": _df([10.0, 11.0]),   # up
        "B": _df([10.0, 9.0]),    # down
        "C": _df([10.0, 12.0]),   # up
    }
    adv, dec, net = pillars.advance_decline(bars)
    assert (adv, dec, net) == (2, 1, 1)


def test_ad_divergence_true_when_index_high_but_breadth_not():
    n = C.HL_LOOKBACK + 5
    highs = np.linspace(400, 500, n)              # SPY at a fresh 52w high
    spy = _df(np.linspace(400, 499, n), highs=highs)
    # cumulative A/D today is BELOW its recent history → not a new high → divergence
    hist = list(range(1000, 1000 + n))            # ascending history, peak ~ 1000+n
    assert pillars.ad_divergence(spy, ad_cumulative_today=500, ad_cumulative_history=hist) is True


def test_ad_divergence_false_when_breadth_confirms():
    n = C.HL_LOOKBACK + 5
    highs = np.linspace(400, 500, n)
    spy = _df(np.linspace(400, 499, n), highs=highs)
    hist = list(range(0, n))
    assert pillars.ad_divergence(spy, ad_cumulative_today=10_000, ad_cumulative_history=hist) is False


# ── Pillar 5: VIX ───────────────────────────────────────────────────────────
def test_vix_read_and_bands():
    rising = pd.Series([12, 13, 14, 15, 16, 17, 18, 19, 20, 25])   # last > sma10
    v = pillars.vix_read(rising)
    assert v["close"] == 25 and v["rising"] is True and v["band"] == "ELEVATED"
    assert pillars.vix_band(10) == "CALM" and pillars.vix_band(35) == "HIGH"


def test_vix_read_none_on_empty():
    assert pillars.vix_read(None) is None
    assert pillars.vix_read(pd.Series([], dtype=float)) is None


# ── Regime resolver ─────────────────────────────────────────────────────────
def _confirmed_inputs():
    return dict(dd_max=1, net_nhnl=50, pct_above_50=70.0, pct_above_200=65.0, ad_divergence=False)


def test_regime_confirmed_uptrend():
    assert regime.resolve(**_confirmed_inputs()) == C.CONFIRMED_UPTREND


def test_regime_correction_paths():
    assert regime.resolve(**{**_confirmed_inputs(), "dd_max": 6}) == C.CORRECTION
    assert regime.resolve(**{**_confirmed_inputs(), "pct_above_200": 35.0}) == C.CORRECTION
    assert regime.resolve(**{**_confirmed_inputs(), "net_nhnl": -5, "pct_above_50": 35.0}) == C.CORRECTION


def test_regime_under_pressure_paths():
    assert regime.resolve(**{**_confirmed_inputs(), "dd_max": 5}) == C.UNDER_PRESSURE
    assert regime.resolve(**{**_confirmed_inputs(), "ad_divergence": True}) == C.UNDER_PRESSURE
    assert regime.resolve(**{**_confirmed_inputs(), "pct_above_50": 45.0}) == C.UNDER_PRESSURE
    assert regime.resolve(**{**_confirmed_inputs(), "net_nhnl": -1}) == C.UNDER_PRESSURE


def test_vix_soft_confirmer_only_under_pressure_never_correction():
    # High rising VIX downgrades a clean tape to UNDER_PRESSURE, never CORRECTION.
    out = regime.resolve(**{**_confirmed_inputs(), "vix_level": 35.0, "vix_rising": True})
    assert out == C.UNDER_PRESSURE


def test_vix_boundary_soft_downgrade():
    out = regime.resolve(**{**_confirmed_inputs(), "dd_max": 4, "vix_level": 26.0, "vix_rising": True})
    assert out == C.UNDER_PRESSURE
    # same dd==4 but calm VIX stays confirmed
    out2 = regime.resolve(**{**_confirmed_inputs(), "dd_max": 4, "vix_level": 14.0, "vix_rising": False})
    assert out2 == C.CONFIRMED_UPTREND


def test_vix_null_fallback_still_resolves_from_pillars():
    # No VIX at all → regime still computes from pillars 1-4.
    assert regime.resolve(**_confirmed_inputs(), vix_level=None, vix_rising=None) == C.CONFIRMED_UPTREND
    assert regime.resolve(**{**_confirmed_inputs(), "dd_max": 6}, vix_level=None, vix_rising=None) == C.CORRECTION


# ── Guidance ────────────────────────────────────────────────────────────────
def test_summary_line():
    row = {"regime": C.UNDER_PRESSURE, "dd_count_spy": 5, "dd_count_qqq": 4,
           "stall_count_spy": 0, "stall_count_qqq": 0, "pct_above_50": 60.0,
           "net_nhnl": 22, "vix_band": "NORMAL", "vix_rising": False,
           "ad_divergence": False, "breadth_thrust": False}
    line = guidance.summary_line(row).lower()
    assert "distribution days" in line and "institutions selling" in line
    assert "healthy breadth" in line and "be selective" in line and line.endswith(".")
    # correction + poor breadth → defense stance
    row2 = {"regime": C.CORRECTION, "dd_count_spy": 7, "dd_count_qqq": 6,
            "pct_above_50": 30.0, "net_nhnl": -50, "vix_band": "HIGH", "vix_rising": True}
    line2 = guidance.summary_line(row2).lower()
    assert "poor breadth" in line2 and "defense" in line2


def test_guidance_build_has_disclaimer_and_vix_line():
    g = guidance.build(C.CORRECTION, "HIGH", True)
    assert "not financial advice" in g["disclaimer"]
    assert g["title"] == "Correction" and g["headline"]
    assert "high" in g["vix_line"].lower()
    # VIX null → unavailable line
    assert "unavailable" in guidance.build(C.CONFIRMED_UPTREND, None, None)["vix_line"].lower()
