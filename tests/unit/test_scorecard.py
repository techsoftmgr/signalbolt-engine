"""
Unit tests — engine/scorecard.py (realized-edge / expectancy aggregation).

Covers the thing that matters most: win rate alone is NOT profitability. A
high-win-rate segment with a bad payoff must still surface a thin/negative
expectancy and a CUT/WATCH verdict.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import scorecard as sc


def _row(pct, result="win", detector="SMC", strat="day_trade", regime="bull"):
    return {"result_pct": pct, "result": result,
            "score_breakdown": {"detector_source": detector},
            "strategy_type": strat, "regime_type": regime}


class TestExpectancyMath:
    def test_high_winrate_low_payoff_is_thin(self):
        # 80% win rate but losers 3x the winners → expectancy barely positive gross,
        # negative AFTER costs. The whole point: win rate hides this.
        rows = [_row(1.0) for _ in range(8)] + [_row(-3.0, result="loss") for _ in range(2)]
        res = sc.compute(rows, cost_pct=0.10)
        p = res["portfolio"]
        assert p["win_rate"] == 80.0
        assert p["avg_win"] == 1.0 and p["avg_loss"] == -3.0
        assert p["payoff"] == 0.33
        assert p["expectancy_gross"] == 0.2          # (8*1 - 2*3)/10
        assert p["expectancy_net"] == 0.1            # 0.2 - 0.10

    def test_high_winrate_good_payoff_keeps(self):
        rows = [_row(2.0) for _ in range(8)] + [_row(-1.5, result="loss") for _ in range(2)]
        res = sc.compute(rows, cost_pct=0.10)
        p = res["portfolio"]
        assert p["win_rate"] == 80.0
        assert p["expectancy_gross"] == 1.3          # (16 - 3)/10
        assert p["expectancy_net"] == 1.2

    def test_worst_loss_and_best_win_tail(self):
        rows = [_row(2.0), _row(5.0), _row(-1.0, result="loss"), _row(-9.0, result="loss")]
        p = sc.compute(rows)["portfolio"]
        assert p["worst_loss"] == -9.0 and p["best_win"] == 5.0


class TestMoneyMade:
    def test_low_winrate_big_payoff_makes_more_total_money(self):
        # The user's exact point: breakdown wins MORE often but breakout makes MORE money.
        breakdown = [_row(0.5, detector="BREAKDOWN") for _ in range(8)] + \
                    [_row(-0.5, result="loss", detector="BREAKDOWN") for _ in range(2)]   # 80% win
        breakout  = [_row(6.0, detector="BREAKOUT") for _ in range(5)] + \
                    [_row(-1.0, result="loss", detector="BREAKOUT") for _ in range(5)]    # 50% win
        res = sc.compute(breakdown + breakout, cost_pct=0.0)
        by = {s["detector"]: s for s in res["segments"]}
        assert by["BREAKDOWN"]["win_rate"] == 80.0
        assert by["BREAKOUT"]["win_rate"] == 50.0
        # Breakdown total = 8*0.5 - 2*0.5 = 3.0% ; Breakout = 5*6 - 5*1 = 25.0%
        assert by["BREAKDOWN"]["total_return_pct"] == 3.0
        assert by["BREAKOUT"]["total_return_pct"] == 25.0
        assert by["BREAKOUT"]["net_total_pct"] > by["BREAKDOWN"]["net_total_pct"]   # more money
        # Ranked by money → breakout first.
        assert res["segments"][0]["detector"] == "BREAKOUT"

    def test_cash_view_normalized_to_notional(self):
        rows = [_row(10.0) for _ in range(1)]   # +10% on one trade
        res = sc.compute(rows, cost_pct=0.0, notional=1000.0)
        p = res["portfolio"]
        assert p["total_return_pct"] == 10.0
        assert p["pnl_per_notional"] == 100.0    # 10% of $1,000
        assert res["notional"] == 1000.0

    def test_profit_share_sums_to_100(self):
        rows = [_row(3.0, detector="A"), _row(1.0, detector="B")]
        res = sc.compute(rows, cost_pct=0.0)
        shares = {s["detector"]: s["profit_share"] for s in res["segments"]}
        assert shares["A"] == 75.0 and shares["B"] == 25.0


class TestVerdict:
    def _many(self, pct, result, n, **kw):
        return [_row(pct, result=result, **kw) for _ in range(n)]

    def test_keep_when_net_positive(self):
        rows = self._many(2.0, "win", 16)
        seg = sc.compute(rows)["segments"][0]
        assert seg["verdict"] == "KEEP"

    def test_cut_when_net_negative(self):
        rows = self._many(-1.0, "loss", 20)
        seg = sc.compute(rows)["segments"][0]
        assert seg["verdict"] == "CUT"

    def test_low_sample_is_watch(self):
        rows = self._many(2.0, "win", 5)        # n < 15
        seg = sc.compute(rows)["segments"][0]
        assert seg["verdict"] == "WATCH" and "low sample" in seg["reason"]


class TestGrouping:
    def test_group_by_regime_splits(self):
        rows = ([_row(2.0, regime="bull") for _ in range(3)] +
                [_row(-2.0, result="loss", regime="bear") for _ in range(3)])
        res = sc.compute(rows, group_by="regime")
        labels = {s["regime"]: s for s in res["segments"]}
        assert set(labels) == {"bull", "bear"}
        assert labels["bull"]["expectancy_gross"] == 2.0
        assert labels["bear"]["expectancy_gross"] == -2.0
        # Portfolio nets to 0 gross (3*2 - 3*2)/6
        assert res["portfolio"]["expectancy_gross"] == 0.0

    def test_detector_regime_isolates_segments(self):
        rows = [_row(1.0, detector="SMC", regime="bull"),
                _row(1.0, detector="MOMENTUM", regime="bull"),
                _row(1.0, detector="SMC", regime="bear")]
        res = sc.compute(rows, group_by="detector_regime")
        assert len(res["segments"]) == 3

    def test_default_groups_detector_strategy(self):
        rows = [_row(1.0, detector="SMC", strat="day_trade"),
                _row(1.0, detector="SMC", strat="swing_trade")]
        res = sc.compute(rows, group_by="detector")
        assert len(res["segments"]) == 2


class TestRobustness:
    def test_skips_none_pct(self):
        rows = [_row(None), _row(2.0)]
        assert sc.compute(rows)["portfolio"]["n"] == 1

    def test_win_inferred_from_positive_pct_when_result_missing(self):
        rows = [_row(1.5, result=None), _row(-1.0, result=None)]
        p = sc.compute(rows)["portfolio"]
        assert p["win_rate"] == 50.0

    def test_empty_rows(self):
        res = sc.compute([])
        assert res["segments"] == [] and res["portfolio"]["n"] == 0

    def test_invalid_group_by_falls_back_to_detector(self):
        res = sc.compute([_row(1.0)], group_by="nonsense")
        assert res["group_by"] == "detector"
