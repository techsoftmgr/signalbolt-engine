"""Unit tests — trend_ride_scorecard.summarize: RODE vs not, trend_break vs baseline, give-back."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import trend_ride_scorecard as sc


def _row(result_pct, *, rode=False, reason="stop_hit", mfe=None,
         strategy="breakout", tf="1Day", result=None, ticker="X"):
    bd = {}
    if rode:
        bd["trend_ride_ever"] = True
    if mfe is not None:
        bd["mfe_pct"] = mfe
    return {
        "ticker": ticker, "direction": "LONG", "strategy_type": strategy, "timeframe": tf,
        "result": result or ("win" if result_pct > 0 else "loss"),
        "result_pct": result_pct, "closed_reason": reason, "score_breakdown": bd,
    }


def test_segments_rode_vs_did_not_and_ignores_non_swings():
    rows = [
        _row(6.0, rode=True,  reason="trend_break"),     # rode, big win
        _row(8.0, rode=True,  reason="target_hit"),      # rode, win
        _row(-2.0, rode=False, reason="structure_reversal"),  # swing, did not ride, early-exit loss
        _row(-1.0, rode=False, reason="market_close"),        # swing, did not ride, early-exit loss
        # non-swing intraday — must be ignored entirely
        _row(5.0, rode=False, reason="target_hit", strategy="day_trade", tf="15m"),
    ]
    out = sc.summarize(rows)
    assert out["counts"]["swings_total"] == 4         # the 15m day_trade excluded
    assert out["counts"]["rode"] == 2
    assert out["counts"]["did_not_ride"] == 2
    assert out["rode"]["n"] == 2 and out["rode"]["win_pct"] == 100.0
    assert out["did_not_ride"]["n"] == 2 and out["did_not_ride"]["win_pct"] == 0.0


def test_trend_break_and_early_baseline_isolated():
    rows = [
        _row(5.0, rode=True, reason="trend_break"),
        _row(7.0, rode=True, reason="trend_break"),
        _row(-1.2, rode=False, reason="structure_reversal"),
        _row(-0.6, rode=False, reason="market_close"),
        _row(2.0, rode=False, reason="target_hit"),     # non-early exit, not in baseline
    ]
    out = sc.summarize(rows)
    assert out["trend_break"]["n"] == 2 and out["trend_break"]["exp"] == 6.0
    # baseline = the two early exits among non-riders only
    assert out["early_exit_baseline"]["n"] == 2
    assert out["counts"]["structure_reversal_on_swings"] == 1


def test_gave_it_back_detection():
    rows = [
        _row(0.1, rode=True, reason="trend_break", mfe=5.0),   # peak +5% but realized +0.1% → gave back
        _row(-1.0, rode=True, reason="stop_hit",   mfe=4.0),   # peak +4% then ended -1% → gave back
        _row(6.0, rode=True, reason="target_hit",  mfe=6.5),   # kept it → NOT give-back
    ]
    out = sc.summarize(rows)
    assert out["gave_back"]["n"] == 2
    assert "X" in out["gave_back"]["tickers"]


def test_empty_is_safe():
    out = sc.summarize([])
    assert out["counts"]["swings_total"] == 0
    assert out["rode"]["n"] == 0 and out["rode"]["pf"] == 0.0
