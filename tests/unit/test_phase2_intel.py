"""Unit tests — Phase 2 intelligence modules (community/watchlist/position) pure
scoring. Additive; existing behavior untouched."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine.phase2 import community_intel as ci
from engine.phase2 import watchlist_intel as wi
from engine.phase2 import position_coach as pc


def test_community_hype_reality_real_momentum():
    h = ci._hype(velocity_pct=80, mentions=400)        # spiking + lots of mentions
    r = ci._reality(ret_5d=8, vol_ratio=1.8, above_ma=True)
    assert h >= 60 and r >= 60
    assert ci._verdict(h, r) == "REAL_MOMENTUM"
    assert "confirmed" in ci._explain(h, r).lower()


def test_community_pump_risk():
    h = ci._hype(velocity_pct=120, mentions=300)       # loud
    r = ci._reality(ret_5d=-4, vol_ratio=0.9, above_ma=False)  # tape says no
    assert h >= 55 and r < 30
    assert ci._verdict(h, r) == "PUMP_RISK"


def test_watchlist_priority_ranks_events_high():
    p_event = wi._priority(rel_strength=1, trend_change=True, earnings_soon=True, near_level=True)
    p_quiet = wi._priority(rel_strength=0.5, trend_change=False, earnings_soon=False, near_level=False)
    assert p_event > p_quiet and p_event >= 70
    assert "earnings" in wi._why(1, "up", True, True, True)


def test_position_risk_and_vol_levels():
    assert pc._risk_level(0, 12) == "HIGH"
    assert pc._risk_level(0, 4) == "CONTAINED"
    assert pc._vol_level(5) == "HIGH"
    assert pc._vol_level(1) == "NORMAL"


def test_position_status_text_mentions_earnings():
    t = pc._status_text("constructive", "ELEVATED", earnings_soon=True,
                        near_support=True, near_resistance=False)
    assert "earnings" in t.lower() and "support" in t.lower()
