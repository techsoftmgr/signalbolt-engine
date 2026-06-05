"""
Unit tests — scorecard conviction grouping (detector × confidence tier), so we
can see whether high-conviction signals actually beat low-conviction ones.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import scorecard


def test_conviction_tier_bands():
    f = scorecard._conviction_tier
    assert f(95) == "A+ (90+)"
    assert f(85) == "A (80-89)"
    assert f(72) == "B+ (70-79)"
    assert f(60) == "B (60-69)"
    assert f(55) == "C (<60)"
    assert f(None) == "?"


def test_detector_conviction_grouping_splits_tiers():
    rows = [
        {"result": "win",  "result_pct": 2.0, "confidence_score": 85,
         "score_breakdown": {"detector_source": "BREAKOUT"}, "strategy_type": "breakout"},
        {"result": "loss", "result_pct": -3.0, "confidence_score": 82,
         "score_breakdown": {"detector_source": "BREAKOUT"}, "strategy_type": "breakout"},
        {"result": "win",  "result_pct": 1.0, "confidence_score": 62,
         "score_breakdown": {"detector_source": "BREAKOUT"}, "strategy_type": "breakout"},
    ]
    res = scorecard.compute(rows, group_by="detector_conviction", min_n=1)
    labels = {s["label"] for s in res["segments"]}
    # Same detector, two different conviction tiers → two segments
    assert any("A (80-89)" in s["conviction"] for s in res["segments"])
    assert any("B (60-69)" in s["conviction"] for s in res["segments"])
    assert len(res["segments"]) == 2
    # each segment carries the detector + conviction in its fields
    for s in res["segments"]:
        assert s["detector"] == "BREAKOUT" and s["conviction"]


def test_unknown_group_by_falls_back_to_detector():
    rows = [{"result": "win", "result_pct": 1.0, "confidence_score": 70,
             "score_breakdown": {"detector_source": "SMC"}, "strategy_type": "day_trade"}]
    res = scorecard.compute(rows, group_by="bogus", min_n=1)
    assert res["group_by"] == "detector"
