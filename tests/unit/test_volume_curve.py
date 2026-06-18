"""Unit tests — volume_curve: ET-session-keyed relvol (overnight reset fix) + the curve."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime
import pandas as pd
from engine import volume_curve as vc


def _rth_df(date="2026-06-17", per_bar_vol=10_000):
    """A full RTH session of 5-min bars (13:30–20:00 UTC = 9:30–4:00 ET) for `date`,
    tz-aware UTC index — mirrors how alpaca_client.get_bars returns intraday bars."""
    idx = pd.date_range(f"{date} 13:30", f"{date} 19:55", freq="5min", tz="UTC")
    n = len(idx)
    return pd.DataFrame({"volume": [per_bar_vol] * n,
                         "high": [1.0]*n, "low": [1.0]*n, "close": [1.0]*n}, index=idx)


def test_curve_caps_at_close_and_floors_at_open():
    assert vc.expected_volume_fraction(390) == 1.0
    assert vc.expected_volume_fraction(500) == 1.0
    assert vc.expected_volume_fraction(0) == vc._VOL_CURVE[1][1]   # floored
    assert 0.30 < vc.expected_volume_fraction(60) < 0.35           # ~0.306 at 60 min


def test_overnight_does_NOT_reset_to_1x():
    """THE BUG: at 12:30 AM ET the UTC date is the next day; the old UTC-date filter found
    no bars → relvol fell to a flat 1.0x. ET-session keying must carry the realized ratio."""
    df = _rth_df("2026-06-17", per_bar_vol=10_000)
    session_vol = df["volume"].sum()           # 78 bars * 10k = 780k
    avg_vol = 390_000                          # → realized ratio = 2.0x
    now = datetime(2026, 6, 18, 0, 30, tzinfo=vc._ET)   # 12:30 AM ET, next UTC day
    rv = vc.session_relvol(df, avg_vol, now=now)
    assert abs(rv - 2.0) < 1e-6                # realized 2.0x carried, NOT 1.0
    assert rv != 1.0


def test_live_rth_projects_via_curve():
    """Mid-session: volume-so-far is projected to a full day via the front-loaded curve."""
    # bars only up to 11:00 ET (90 min in) — partial session
    idx = pd.date_range("2026-06-17 13:30", "2026-06-17 14:55", freq="5min", tz="UTC")
    df = pd.DataFrame({"volume": [10_000]*len(idx)}, index=idx)
    sofar = df["volume"].sum()
    now = datetime(2026, 6, 17, 11, 0, tzinfo=vc._ET)   # 90 min after open
    avg_vol = 500_000
    rv = vc.session_relvol(df, avg_vol, now=now)
    frac = vc.expected_volume_fraction(90)              # ~0.387
    assert abs(rv - (sofar / frac) / avg_vol) < 1e-6
    # projection must inflate vs the raw (un-projected) ratio
    assert rv > (sofar / avg_vol)


def test_after_close_same_day_is_realized_not_projected():
    df = _rth_df("2026-06-17", per_bar_vol=10_000)
    avg_vol = df["volume"].sum()                        # → 1.0 realized... use a clean ratio
    now = datetime(2026, 6, 17, 17, 30, tzinfo=vc._ET)  # 5:30 PM ET (after close, same UTC day)
    rv = vc.session_relvol(df, avg_vol, now=now)
    assert abs(rv - 1.0) < 1e-6                          # realized = sessionvol/avg, no projection
    # and with a smaller avg it should exceed 1 (proves it's realized ratio, not the fallback)
    assert vc.session_relvol(df, avg_vol / 2, now=now) > 1.9


def test_weekend_carries_last_session():
    df = _rth_df("2026-06-19", per_bar_vol=10_000)      # Friday session
    avg_vol = df["volume"].sum() / 1.5                  # realized 1.5x
    now = datetime(2026, 6, 20, 12, 0, tzinfo=vc._ET)   # Saturday noon
    rv = vc.session_relvol(df, avg_vol, now=now)
    assert abs(rv - 1.5) < 1e-6                          # Friday's reading carried, not 1.0


def test_defaults_to_1x_only_when_no_data():
    now = datetime(2026, 6, 17, 11, 0, tzinfo=vc._ET)
    assert vc.session_relvol(None, 100, now=now) == 1.0
    assert vc.session_relvol(_rth_df(), 0, now=now) == 1.0          # no avg
    empty = pd.DataFrame({"volume": []}, index=pd.DatetimeIndex([], tz="UTC"))
    assert vc.session_relvol(empty, 100, now=now) == 1.0


def test_latest_session_bars_picks_most_recent_et_date():
    a = _rth_df("2026-06-16", per_bar_vol=1)
    b = _rth_df("2026-06-17", per_bar_vol=2)
    df = pd.concat([a, b])
    out = vc.latest_session_bars(df)
    assert (out["volume"] == 2).all()                   # only the 06-17 session
