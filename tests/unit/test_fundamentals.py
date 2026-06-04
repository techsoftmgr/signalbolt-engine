"""
Unit tests — engine/fundamentals.py XBRL normalization.

Centerpiece: prove the prototype's bug is fixed — when a partial/segment revenue
tag and the real total-revenue tag both exist, compute_metrics() uses the TOTAL
(largest annual per FY), so net margin is sane (not 148%/600%).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from engine import fundamentals as fnd


def _flow(val, fy, end, frame=None):
    e = {"val": val, "fy": fy, "fp": "FY", "form": "10-K", "end": end}
    if frame:
        e["frame"] = frame
    return e


def _point(val, end):
    return {"val": val, "end": end}


def _facts(usgaap: dict) -> dict:
    return {"facts": {"us-gaap": {t: {"units": {"USD": entries}} for t, entries in usgaap.items()}}}


class TestRevenueNormalization:
    def test_picks_largest_revenue_tag(self):
        """Partial 'Revenues' (5k) vs real total (100k) → use the total → margin sane."""
        f = _facts({
            "Revenues": [_flow(5_000, 2024, "2024-12-31")],                                   # partial
            "RevenueFromContractWithCustomerExcludingAssessedTax": [_flow(100_000, 2024, "2024-12-31")],
            "NetIncomeLoss": [_flow(30_000, 2024, "2024-12-31")],
            "StockholdersEquity": [_point(50_000, "2024-12-31")],
        })
        m = fnd.compute_metrics(f)
        assert m["revenue_latest"] == 100_000          # total, not the 5k partial
        assert m["net_margin"] == pytest.approx(30.0)  # NOT 600%
        assert m["roe"] == pytest.approx(60.0)

    def test_implausible_margin_dropped(self):
        """Only a tiny revenue tag (margin would be >80%) → drop net_margin."""
        f = _facts({
            "Revenues": [_flow(1_000, 2024, "2024-12-31")],
            "NetIncomeLoss": [_flow(30_000, 2024, "2024-12-31")],
        })
        m = fnd.compute_metrics(f)
        assert m["net_margin"] is None   # sanity guard caught the bad revenue

    def test_revenue_growth_pairs_fiscal_years(self):
        f = _facts({
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                _flow(110_000, 2024, "2024-12-31"),
                _flow(100_000, 2023, "2023-12-31"),
            ],
        })
        m = fnd.compute_metrics(f)
        assert m["revenue_growth"] == pytest.approx(10.0)


class TestBalanceAndCashflow:
    def test_debt_is_lt_plus_current(self):
        f = _facts({
            "LongTermDebtNoncurrent": [_point(40_000, "2024-12-31")],
            "LongTermDebtCurrent": [_point(10_000, "2024-12-31")],
            "StockholdersEquity": [_point(50_000, "2024-12-31")],
        })
        m = fnd.compute_metrics(f)
        assert m["debt"] == 50_000
        assert m["debt_to_equity"] == pytest.approx(1.0)

    def test_fcf(self):
        f = _facts({
            "NetCashProvidedByUsedInOperatingActivities": [_flow(50_000, 2024, "2024-12-31")],
            "PaymentsToAcquirePropertyPlantAndEquipment": [_flow(20_000, 2024, "2024-12-31")],
        })
        m = fnd.compute_metrics(f)
        assert m["fcf"] == 30_000
        assert m["fcf_positive"] is True

    def test_empty_facts_safe(self):
        m = fnd.compute_metrics({})
        assert m["net_margin"] is None and m["fcf_positive"] is False
        assert fnd.quality_score(m) == 0


class TestQualityScore:
    def test_perfect_quality(self):
        m = {"net_margin": 30, "roe": 25, "debt_to_equity": 0.4, "revenue_growth": 12, "fcf_positive": True}
        assert fnd.quality_score(m) == 5

    def test_leveraged_unprofitable_scores_low(self):
        m = {"net_margin": -2, "roe": -1, "debt_to_equity": 7.8, "revenue_growth": -3, "fcf_positive": False}
        assert fnd.quality_score(m) == 0

    def test_missing_metrics_count_as_fail(self):
        m = {"net_margin": None, "roe": 20, "debt_to_equity": None, "revenue_growth": None, "fcf_positive": True}
        assert fnd.quality_score(m) == 2   # roe + fcf only


class TestUniverse:
    def test_universe_is_curated_large_set(self):
        assert len(fnd.QUALITY_UNIVERSE) >= 100
        assert "AAPL" in fnd.QUALITY_UNIVERSE and "JPM" in fnd.QUALITY_UNIVERSE
        assert len(set(fnd.QUALITY_UNIVERSE)) == len(fnd.QUALITY_UNIVERSE)  # no dups
