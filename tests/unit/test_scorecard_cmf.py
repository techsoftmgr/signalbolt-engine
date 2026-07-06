"""
scorecard.py — money-flow (CMF) segmentation.

Every fired signal is stamped with its fire-time cmfState (via signal_telemetry).
group_by='cmf' buckets closed signals by that state so we can MEASURE whether
signals fired during accumulation beat those fired during distribution — the
measure-first gate before wiring CMF into firing.
"""
from engine import scorecard as sc


def _row(pct, result="win", cmf_state="accumulation", detector="SMC"):
    return {"result_pct": pct, "result": result, "direction": "LONG",
            "strategy_type": "swing_trade", "regime_type": "RANGING",
            "score_breakdown": {"detector_source": detector, "cmfState": cmf_state}}


def _seg(res, label_contains):
    return next((s for s in res["segments"] if label_contains in s["label"]), None)


def test_cmf_grouping_splits_by_flow_state():
    rows = ([_row(2.0, cmf_state="accumulation") for _ in range(4)] +
            [_row(-1.0, result="loss", cmf_state="distribution") for _ in range(3)])
    res = sc.compute(rows, group_by="cmf", min_n=1)
    accum = _seg(res, "flow:accumulation")
    distr = _seg(res, "flow:distribution")
    assert accum is not None and distr is not None
    assert accum["n"] == 4 and accum["win_rate"] == 100.0
    assert distr["n"] == 3 and distr["win_rate"] == 0.0


def test_detector_cmf_crosses_detector_with_flow():
    rows = [_row(1.0, cmf_state="accumulation"), _row(1.0, cmf_state="distribution")]
    res = sc.compute(rows, group_by="detector_cmf", min_n=1)
    labels = [s["label"] for s in res["segments"]]
    assert any("SMC" in l and "flow:accumulation" in l for l in labels)
    assert any("SMC" in l and "flow:distribution" in l for l in labels)


def test_missing_cmf_falls_back_to_unknown():
    rows = [{"result_pct": 1.0, "result": "win", "direction": "LONG",
             "strategy_type": "day_trade", "score_breakdown": {"detector_source": "SMC"}}]
    res = sc.compute(rows, group_by="cmf", min_n=1)
    assert _seg(res, "flow:unknown") is not None
