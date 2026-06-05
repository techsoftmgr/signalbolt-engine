"""Unit tests — alpha.position_alpha (excess vs SPY, direction-aware) + the
scorecard avg_alpha / market_beat_rate aggregation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine import alpha, scorecard


def test_long_alpha_excess_over_spy():
    # LONG +10%, SPY 100→105 (+5%) → alpha +5
    a = alpha.position_alpha("LONG", 10.0, 100, 105)
    assert a["benchmark_return_pct"] == 5.0 and a["alpha_pct"] == 5.0


def test_long_rode_the_tape_zero_alpha():
    # LONG +5% but SPY also +5% → no alpha (just beta)
    a = alpha.position_alpha("LONG", 5.0, 100, 105)
    assert a["alpha_pct"] == 0.0


def test_short_benchmark_is_negative_spy():
    # SHORT +3% while SPY fell 100→96 (−4%): being short the market earned +4,
    # so the position UNDER-performed its market exposure → alpha −1
    a = alpha.position_alpha("SHORT", 3.0, 100, 96)
    assert a["benchmark_return_pct"] == 4.0 and a["alpha_pct"] == -1.0


def test_short_beats_in_up_market():
    # SHORT +2% while SPY ROSE 100→102 (benchmark for a short = −2) → alpha +4
    a = alpha.position_alpha("SHORT", 2.0, 100, 102)
    assert a["benchmark_return_pct"] == -2.0 and a["alpha_pct"] == 4.0


def test_bad_inputs_none():
    assert alpha.position_alpha("LONG", 1.0, 0, 5) is None
    assert alpha.position_alpha("LONG", None, 100, 101) is None


def test_scorecard_alpha_aggregation():
    rows = [
        {"result": "win",  "result_pct": 10.0, "confidence_score": 75, "strategy_type": "swing",
         "score_breakdown": {"detector_source": "TREND_MOMENTUM", "alpha_pct": 5.0}},
        {"result": "win",  "result_pct": 5.0,  "confidence_score": 75, "strategy_type": "swing",
         "score_breakdown": {"detector_source": "TREND_MOMENTUM", "alpha_pct": 0.0}},   # rode tape
        {"result": "loss", "result_pct": -3.0, "confidence_score": 75, "strategy_type": "swing",
         "score_breakdown": {"detector_source": "TREND_MOMENTUM", "alpha_pct": -1.0}},
    ]
    seg = scorecard.compute(rows, group_by="detector", min_n=1)["segments"][0]
    assert seg["alpha_sample"] == 3
    assert seg["avg_alpha"] == round((5 + 0 - 1) / 3, 2)        # 1.33
    assert seg["market_beat_rate"] == round(100 * 1 / 3, 1)     # only the +5 beat → 33.3
