"""
scorecard.py — RS-exemption cohort visibility.

Longs that fired through the regime long-veto on relative strength carry
score_breakdown.rs_exempt. They must (a) be isolable via group_by='rs_exempt',
and (b) surface as their own 'SMC·RSx' line in the normal detector view, so we
can watch the experimental cohort's realized edge vs standard signals.
"""
from engine import scorecard as sc


def _row(pct, result="win", detector="SMC", strat="day_trade", regime="RISK_OFF", rs=False):
    bd = {"detector_source": detector}
    if rs:
        bd["rs_exempt"] = {"regime": regime, "rs_vs_spy_pct": 12.3}
    return {"result_pct": pct, "result": result, "score_breakdown": bd,
            "strategy_type": strat, "regime_type": regime, "direction": "LONG"}


def _seg(res, label_contains):
    return next((s for s in res["segments"] if label_contains in s["label"]), None)


def test_rs_exempt_grouping_splits_cohorts():
    rows = ([_row(2.0, rs=True) for _ in range(3)] +
            [_row(-1.0, result="loss", rs=True)] +
            [_row(1.0) for _ in range(5)])
    res = sc.compute(rows, group_by="rs_exempt", min_n=1)
    exempt = _seg(res, "RS-exempt")
    std    = _seg(res, "Standard")
    assert exempt is not None and std is not None
    assert exempt["n"] == 4 and std["n"] == 5
    assert exempt["win_rate"] == 75.0


def test_rs_exempt_surfaces_as_own_detector_line():
    rows = [_row(2.0, rs=True), _row(1.0, rs=False)]
    res = sc.compute(rows, group_by="detector", min_n=1)
    labels = [s["label"] for s in res["segments"]]
    assert any("SMC·RSx" in l for l in labels)   # exempt cohort isolated
    assert any(l for l in labels if "SMC" in l and "RSx" not in l)  # standard SMC separate


def test_standard_signals_unaffected_when_no_rs_exempt():
    rows = [_row(1.0), _row(-1.0, result="loss")]
    res = sc.compute(rows, group_by="rs_exempt", min_n=1)
    # No RS-exempt rows → only the Standard cohort exists.
    assert _seg(res, "RS-exempt") is None
    assert _seg(res, "Standard") is not None
