"""Unit tests — IPO calendar. Offline (mocked fetch)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import ipo_calendar as ipo


def test_get_ipo_calendar_splits_and_maps(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "x")

    def fake_fetch(params):
        # priced query carries listing_date.lt; upcoming carries only .gte
        if "listing_date.lt" in params:
            return [
                {"ticker": "CCC", "issuer_name": "Gamma", "listing_date": "2026-06-10",
                 "final_issue_price": 18.0, "ipo_status": "new"},
                {"ticker": "DDD", "issuer_name": "Delta", "final_issue_price": None, "ipo_status": "new"},
            ]
        return [
            {"ticker": "AAA", "issuer_name": "Alpha", "listing_date": "2026-06-12",
             "lowest_offer_price": 135, "highest_offer_price": 135, "ipo_status": "pending",
             "primary_exchange": "XNAS"},
            {"ticker": "BBB", "issuer_name": "Beta", "listing_date": "2026-06-20",
             "lowest_offer_price": 10, "highest_offer_price": 12, "ipo_status": "pending"},
        ]

    monkeypatch.setattr(ipo, "_fetch", fake_fetch)
    out = ipo.get_ipo_calendar(force=True)

    assert out["available"] is True and out["source"] == "polygon"
    # upcoming: soonest listing first (AAA 6/12 before BBB 6/20)
    assert [u["ticker"] for u in out["upcoming"]] == ["AAA", "BBB"]
    assert out["upcoming"][0]["price_low"] == 135.0 and out["upcoming"][0]["price_high"] == 135.0
    # priced: only rows WITH a finalized issue price (DDD dropped)
    assert len(out["priced"]) == 1
    assert out["priced"][0]["ticker"] == "CCC" and out["priced"][0]["final_price"] == 18.0


def test_get_ipo_calendar_no_key(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setattr(ipo, "_fetch", lambda params: [])
    out = ipo.get_ipo_calendar(force=True)
    assert out["available"] is False and out["source"] == "unavailable"
    assert out["upcoming"] == [] and out["priced"] == []
