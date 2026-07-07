"""
New OHLCV indicators in quant_score_service: ADX (trend strength), TTM Squeeze
(volatility coil), MFI (volume-weighted RSI), ATR/ADR (range + stop width).
Pure math — pin direction/bounds/states + insufficient-data safety.
"""
import numpy as np
import pandas as pd

from engine.quant_score_service import (
    _adx, _adx_state, _squeeze, _mfi, _mfi_state, _atr_adr,
)


def _df(closes, rng=1.0, vol=1_000_000):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": closes, "high": closes + rng, "low": closes - rng,
        "close": closes, "volume": [vol] * len(closes),
    })


# ── ADX ──────────────────────────────────────────────────────────────────────
def test_adx_high_on_clean_trend_low_on_chop():
    trend = _df(np.linspace(50, 150, 60))                    # steady uptrend
    chop  = _df([100 + (2 if i % 2 else -2) for i in range(60)])  # zigzag
    a_trend, a_chop = _adx(trend), _adx(chop)
    assert a_trend is not None and a_chop is not None
    assert a_trend > a_chop
    assert _adx_state(a_trend) == "trending"     # strong trend
    assert _adx_state(a_chop) == "choppy"        # no trend


def test_adx_states_and_none():
    assert _adx_state(30) == "trending" and _adx_state(22) == "developing"
    assert _adx_state(10) == "choppy" and _adx_state(None) == "unknown"
    assert _adx(_df(np.linspace(1, 5, 8))) is None   # too few bars


# ── TTM Squeeze ──────────────────────────────────────────────────────────────
def test_squeeze_returns_valid_state_and_bias():
    up = _df(np.linspace(90, 110, 40))
    state, bias = _squeeze(up)
    assert state in ("on", "fired", "off")
    assert bias == "bull"                         # price above the basis
    dn = _df(np.linspace(110, 90, 40))
    assert _squeeze(dn)[1] == "bear"


def test_squeeze_unknown_on_short_data():
    assert _squeeze(_df([1, 2, 3])) == ("unknown", "flat")


def test_squeeze_ignores_todays_forming_bar():
    # A daily indicator should read the last COMPLETED bar — dropping today's
    # still-forming bar so the state doesn't flicker fired↔coiling intraday.
    from datetime import datetime
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    today = pd.Timestamp(datetime.now(et).date(), tz=et)
    idx = pd.date_range(end=today, periods=41, freq="D", tz=et)   # last row == today
    closes = np.linspace(90, 110, 41)
    df = pd.DataFrame({"open": closes, "high": closes + 1, "low": closes - 1,
                       "close": closes, "volume": [1e6] * 41}, index=idx)
    # Make today's (forming) bar anomalous — it would change the state IF used.
    df.iloc[-1, df.columns.get_loc("high")] = 200.0
    df.iloc[-1, df.columns.get_loc("low")]  = 50.0
    df.iloc[-1, df.columns.get_loc("close")] = 195.0
    assert _squeeze(df) == _squeeze(df.iloc[:-1])   # today's bar was dropped


# ── MFI ──────────────────────────────────────────────────────────────────────
def test_mfi_bounds_and_states():
    up = _df(np.linspace(50, 150, 30))            # every bar up → strong inflow
    dn = _df(np.linspace(150, 50, 30))
    assert _mfi(up) >= 80 and _mfi_state(_mfi(up)) == "overbought"
    assert _mfi(dn) <= 20 and _mfi_state(_mfi(dn)) == "oversold"
    assert _mfi(_df([1, 2, 3])) is None


# ── ATR / ADR ────────────────────────────────────────────────────────────────
def test_atr_adr_positive_and_scales_with_range():
    tight = _df(np.full(30, 100.0), rng=0.5)
    wide  = _df(np.full(30, 100.0), rng=4.0)
    adr_t, stop_t = _atr_adr(tight, 100.0)
    adr_w, stop_w = _atr_adr(wide, 100.0)
    assert adr_t > 0 and stop_t > 0 and adr_w > adr_t and stop_w > stop_t
    assert _atr_adr(_df([1, 2, 3]), 100.0) == (None, None)
