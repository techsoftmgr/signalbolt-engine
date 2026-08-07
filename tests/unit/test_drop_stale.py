"""alpaca_client.drop_stale — exclude delisted/long-halted tickers (frozen bars) from
current-state consumers (scan universe, churn). The EA-taken-private case."""
import pandas as pd
from engine import alpaca_client as ac


def _df(dates, close=100.0):
    idx = pd.to_datetime(dates, utc=True)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1e6}, index=idx)


def test_drop_stale_removes_delisted_keeps_live():
    bars = {
        "SPY":  _df(["2026-08-05", "2026-08-06", "2026-08-07"]),   # the market ref
        "AAPL": _df(["2026-08-05", "2026-08-06", "2026-08-07"]),   # live → kept
        "EA":   _df(["2026-08-04", "2026-08-05", "2026-08-06"]),   # 1d behind → kept (guard is conservative)
        "DEAD": _df(["2026-07-10", "2026-07-14", "2026-07-15"]),   # ~3wk behind → DROPPED (delisted)
    }
    out = ac.drop_stale(bars)
    assert set(out) == {"SPY", "AAPL", "EA"}
    assert "DEAD" not in out


def test_drop_stale_failopen_without_ref():
    bars = {"XYZ": _df(["2026-01-01", "2026-01-02"])}
    assert ac.drop_stale(bars) == bars          # no SPY ref → keep everything (fail-open)
