"""
Unit tests — scorecard exit-quality enrichment (avg MFE / MAE / winner-MAE /
give-back / time-to-peak / mae-before-mfe). These are the stats that turn the
keep/cut scorecard into the profit-lock + min-stop tuning lens, segmentable by
regime. Also covers signal_monitor._mins_since_entry (the timing source).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import scorecard


def _row(detector, pct, mfe=None, mae=None, t_mfe=None, t_mae=None, result=None):
    sb = {"detector_source": detector}
    if mfe is not None: sb["mfe_pct"] = mfe
    if mae is not None: sb["mae_pct"] = mae
    if t_mfe is not None: sb["t_mfe_min"] = t_mfe
    if t_mae is not None: sb["t_mae_min"] = t_mae
    res = result if result is not None else ("win" if pct > 0 else "loss")
    return {"result": res, "result_pct": pct, "confidence_score": 75,
            "score_breakdown": sb, "strategy_type": "swing"}


def test_exit_quality_aggregates():
    rows = [
        # winner: ran +20% peak, realized +12% (gave back 8), worst dip -2%
        _row("TREND_MOMENTUM", 12.0, mfe=20.0, mae=-2.0, t_mfe=300, t_mae=30),
        # winner: peak +8, realized +6 (gave back 2), worst dip -4%
        _row("TREND_MOMENTUM", 6.0,  mfe=8.0,  mae=-4.0, t_mfe=120, t_mae=200),
        # loser: peak +1, realized -5 (no give-back vs realized = 6), worst -6%
        _row("TREND_MOMENTUM", -5.0, mfe=1.0,  mae=-6.0, t_mfe=15,  t_mae=240),
    ]
    seg = scorecard.compute(rows, group_by="detector", min_n=1)["segments"][0]
    assert seg["mfe_sample"] == 3
    assert seg["avg_mfe"] == round((20 + 8 + 1) / 3, 2)            # 9.67
    assert seg["avg_mae"] == round((-2 + -4 + -6) / 3, 2)          # -4.0
    assert seg["winner_mae"] == round((-2 + -4) / 2, 2)            # -3.0  (winners only)
    # give-back = max(0, mfe - realized): 8, 2, 6  → avg 5.33
    assert seg["avg_giveback"] == round((8 + 2 + 6) / 3, 2)
    assert seg["avg_t_mfe_min"] == round((300 + 120 + 15) / 3, 1)  # 145.0
    # mae_before_mfe: row1 30<300 yes, row2 200<120 no, row3 240<15 no → 1/3
    assert seg["timing_sample"] == 3
    assert seg["mae_before_mfe_pct"] == round(100 * 1 / 3, 1)      # 33.3


def test_exit_quality_none_without_breakdown():
    rows = [{"result": "win", "result_pct": 2.0, "confidence_score": 70,
             "score_breakdown": {"detector_source": "SMC"}, "strategy_type": "day_trade"}]
    seg = scorecard.compute(rows, group_by="detector", min_n=1)["segments"][0]
    assert seg["avg_mfe"] is None and seg["winner_mae"] is None
    assert seg["mae_before_mfe_pct"] is None and seg["mfe_sample"] == 0


def test_exit_quality_segments_by_regime():
    rows = [
        _row("BREAKDOWN", 9.0,  mfe=10.0, mae=-1.0),
        _row("BREAKDOWN", -4.0, mfe=2.0,  mae=-5.0),
    ]
    rows[0]["regime_type"] = "PANIC"
    rows[1]["regime_type"] = "TRENDING_BULL"
    res = scorecard.compute(rows, group_by="regime", min_n=1)
    by = {s["regime"]: s for s in res["segments"]}
    assert by["PANIC"]["avg_mfe"] == 10.0 and by["PANIC"]["winner_mae"] == -1.0
    assert by["TRENDING_BULL"]["winner_mae"] is None      # no winners in that regime cell


def test_mins_since_entry():
    from engine.signal_monitor import _mins_since_entry
    import datetime as dt
    assert _mins_since_entry(None) is None
    assert _mins_since_entry("garbage") is None
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=60)).isoformat()
    m = _mins_since_entry(past)
    assert m is not None and 59.0 <= m <= 61.0
