"""
Unit tests — churn_service: the high-volume / low-price-move "absorption" read,
its range-position zone tag, and the event (news-gap pin) flag.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime
import pandas as pd

from engine import churn_service as cs

END = "2026-06-15"
TODAY = pd.Timestamp(END).date()


def _df(closes, vols):
    n = len(closes)
    idx = pd.bdate_range(end=END, periods=n)
    return pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": vols}, index=idx)


# ── zone + session fraction ─────────────────────────────────────────────────
def test_classify_zone():
    assert cs.classify_zone(0.10) == "accumulation"
    assert cs.classify_zone(0.50) == "churn"
    assert cs.classify_zone(0.90) == "distribution"


def test_session_fraction():
    assert cs.session_fraction(datetime(2026, 6, 15, 8, 0)) == 1.0     # premarket → no projection
    assert cs.session_fraction(datetime(2026, 6, 15, 17, 0)) == 1.0    # after close → full day
    mid = cs.session_fraction(datetime(2026, 6, 15, 12, 0))            # ~2.5h in
    assert 0.0 < mid < 1.0


# ── _evaluate: qualifies / rejects ──────────────────────────────────────────
def test_evaluate_flat_high_volume_churns():
    # 20-day range ~98–102, prev 100.0, today 100.2 → flat & mid-range
    closes = [98.0, 102.0] * 10 + [100.0, 100.2]
    vols   = [1_000_000] * 21 + [2_000_000]  # today 2× avg
    it = cs._evaluate("FLAT", _df(closes, vols), frac=1.0, today=TODAY)
    assert it is not None
    assert it["relVol"] == 2.0 and abs(it["changePct"]) <= 3.0
    assert it["zone"] == "churn" and it["strong"] is True and it["event"] is False
    assert it["churnScore"] > 0


def test_evaluate_rejects_big_mover():
    closes = [100.0] * 21 + [106.0]          # +6% move — a real move, not absorption
    vols   = [1_000_000] * 21 + [3_000_000]
    assert cs._evaluate("MOV", _df(closes, vols), frac=1.0, today=TODAY) is None


def test_churn_score_ranks_heavy_volume_small_move_highest():
    # ROKU-like: 8× volume, -1.6% (within the 3% ceiling) must outscore a 1.1×/0.0% name
    roku = cs._evaluate("ROKU", _df([100.0] * 21 + [98.4], [1_000_000] * 21 + [8_000_000]), frac=1.0, today=TODAY)
    mild = cs._evaluate("MILD", _df([100.0, 101.0] * 10 + [100.0, 100.0], [1_000_000] * 21 + [1_100_000]), frac=1.0, today=TODAY)
    assert roku is not None and mild is not None
    assert roku["churnScore"] > mild["churnScore"]


def test_evaluate_rejects_low_volume():
    closes = [100.0] * 21 + [100.1]
    vols   = [1_000_000] * 21 + [600_000]    # below average → not churn
    assert cs._evaluate("LOW", _df(closes, vols), frac=1.0, today=TODAY) is None


def test_evaluate_pace_adjusts_volume():
    # mid-session: 1.2M so far at 50% of the day projects to 2.4M → 2.4× avg
    closes = [100.0] * 21 + [100.1]
    vols   = [1_000_000] * 21 + [1_200_000]
    it = cs._evaluate("PACE", _df(closes, vols), frac=0.5, today=TODAY)
    assert it is not None and it["relVol"] == 2.4


def test_evaluate_flags_event_pin_near_highs():
    # ROKU-like: a +20% gap on heavy volume yesterday, pinned flat today near highs
    closes = [100.0] * 19 + [100.0, 120.0, 120.3]
    vols   = [1_000_000] * 19 + [1_000_000, 6_000_000, 3_000_000]
    it = cs._evaluate("ROKU", _df(closes, vols), frac=1.0, today=TODAY)
    assert it is not None
    assert it["event"] is True               # recent gap on heavy volume
    assert it["zone"] == "distribution"      # sitting near the top of its 20-day range
    assert abs(it["changePct"]) <= 1.5       # but flat today (the pin)


def test_evaluate_rejects_stale_data():
    # last bar is far in the past (> _STALE_DAYS) → skip (halted / delisted)
    closes = [100.0] * 22
    vols   = [1_000_000] * 21 + [2_000_000]
    assert cs._evaluate("OLD", _df(closes, vols), frac=1.0, today=pd.Timestamp("2026-07-01").date()) is None


def test_carries_recent_session_overnight():
    # FIX: last bar is the prior session; "today" is the next day (overnight / pre-open).
    # Outside RTH frac=1.0 → the realized read must STILL be returned, not blanked like the
    # old `!= today` guard did from midnight ET until the next open. Updates live at the open
    # when a new forming bar appears.
    closes = [98.0, 102.0] * 10 + [100.0, 100.2]
    vols   = [1_000_000] * 21 + [2_000_000]
    nxt = (pd.Timestamp(END) + pd.Timedelta(days=1)).date()   # day after the last session bar
    it = cs._evaluate("CARRY", _df(closes, vols), frac=1.0, today=nxt)
    assert it is not None and it["relVol"] == 2.0   # carried realized 2.0x, not dropped
