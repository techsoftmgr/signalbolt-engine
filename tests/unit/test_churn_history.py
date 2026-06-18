"""Unit tests — churn_history: forward-return, streak buckets, zone/streak aggregation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import date
import pandas as pd
from engine import churn_history as ch


def _daily(closes, start="2026-06-01"):
    idx = pd.date_range(start=start, periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"close": closes, "high": closes, "low": closes, "volume": [1]*len(closes)}, index=idx)


def test_forward_return_horizon():
    # closes rise 100→105 over 5 sessions after the obs day
    df = _daily([100, 101, 102, 103, 104, 105, 106])
    sd = df.index[0].date()
    fwd = ch._forward_return(df, sd, obs_close=100.0, horizon=5)
    assert abs(fwd - 5.0) < 1e-6           # close[+5]=105 → +5%


def test_forward_return_not_elapsed_is_none():
    df = _daily([100, 101, 102])           # only 3 bars, horizon 5 not reached
    assert ch._forward_return(df, df.index[0].date(), 100.0, 5) is None


def test_forward_return_missing_session_is_none():
    df = _daily([100, 101, 102, 103, 104, 105])
    assert ch._forward_return(df, date(2020, 1, 1), 100.0, 5) is None


def test_streak_bucket():
    assert ch._streak_bucket(1) == "1"
    assert ch._streak_bucket(2) == "2"
    assert ch._streak_bucket(3) == "3+"
    assert ch._streak_bucket(9) == "3+"


def test_aggregate_by_zone_directional_hitrate():
    judged = [
        {"zone": "accumulation", "streak": 1, "fwd": 4.0},   # up → hit
        {"zone": "accumulation", "streak": 2, "fwd": -1.0},  # down → miss
        {"zone": "distribution", "streak": 1, "fwd": -3.0},  # down → hit (expected down)
        {"zone": "distribution", "streak": 3, "fwd": 2.0},   # up → miss
        {"zone": "churn",        "streak": 1, "fwd": 1.0},
    ]
    out = ch._aggregate(judged)
    assert out["n"] == 5 and out["horizonDays"] == ch.HORIZON_DAYS
    acc = out["byZone"]["accumulation"]
    assert acc["n"] == 2 and acc["hitRate"] == 50.0      # 1 of 2 resolved UP
    dist = out["byZone"]["distribution"]
    assert dist["n"] == 2 and dist["hitRate"] == 50.0    # 1 of 2 resolved DOWN
    assert out["byZone"]["churn"]["hitRate"] is None     # no directional bias for churn
    # streak buckets present
    assert out["byStreak"]["1"]["n"] == 3 and out["byStreak"]["2"]["n"] == 1 and out["byStreak"]["3+"]["n"] == 1


def test_aggregate_empty():
    out = ch._aggregate([])
    assert out["n"] == 0
    assert out["byZone"]["accumulation"]["n"] == 0
