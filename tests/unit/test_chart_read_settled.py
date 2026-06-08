"""Unit test — chart_read._settled drops the still-forming bar so the read is a
stable per-period plan (the "moving target" fix). Additive, offline."""
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from engine import chart_read as cr


def _daily_df():
    idx = pd.to_datetime(["2026-06-04 05:00", "2026-06-05 05:00", "2026-06-08 05:00"], utc=True)
    return pd.DataFrame({"open": [99, 100, 101], "high": [101, 102, 103],
                         "low": [98, 99, 100], "close": [100, 101, 102],
                         "volume": [1e6, 1e6, 1e6]}, index=idx)


def test_daily_forming_bar_dropped_intraday():
    df = _daily_df()
    # 10:00 ET on Jun 8 (session in progress) → today's bar is forming → dropped
    out = cr._settled(df, "1Day", now=datetime(2026, 6, 8, 14, 0, tzinfo=timezone.utc))
    assert len(out) == 2
    assert out["close"].iloc[-1] == 101          # anchored to Jun 5 close


def test_daily_bar_kept_after_close():
    df = _daily_df()
    # 5:00 PM ET on Jun 8 (after the 4 PM close) → today's bar is settled → kept
    out = cr._settled(df, "1Day", now=datetime(2026, 6, 8, 21, 0, tzinfo=timezone.utc))
    assert len(out) == 3
    assert out["close"].iloc[-1] == 102


def test_daily_kept_next_day():
    df = _daily_df()
    out = cr._settled(df, "1Day", now=datetime(2026, 6, 9, 14, 0, tzinfo=timezone.utc))
    assert len(out) == 3                          # last bar is no longer "today"


def test_hourly_forming_hour_dropped():
    idx = pd.to_datetime(["2026-06-08 17:00", "2026-06-08 18:00"], utc=True)
    df = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [1, 1]}, index=idx)
    # within the 18:00 UTC hour → that bar is forming → dropped
    out = cr._settled(df, "1Hour", now=datetime(2026, 6, 8, 18, 30, tzinfo=timezone.utc))
    assert len(out) == 1
    # next hour → kept
    out2 = cr._settled(df, "1Hour", now=datetime(2026, 6, 8, 19, 5, tzinfo=timezone.utc))
    assert len(out2) == 2


def test_short_df_unchanged():
    df = _daily_df().iloc[:1]
    assert len(cr._settled(df, "1Day", now=datetime(2026, 6, 8, 14, 0, tzinfo=timezone.utc))) == 1
