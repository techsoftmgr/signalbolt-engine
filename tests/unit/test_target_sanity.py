"""Regression — high-ATR / leveraged names must not produce absurd targets.
KORU (a 3x China ETF, ~25% ATR) blew T2 below zero (T1 $232 / T2 -$159 on a $624
short), so the T1->breakeven profit-lock could never fire. The ATR is now capped
at 8% of price for SL/TP, keeping targets positive + sane."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine import forming_signals, breakdown_signals


def _capped(price, atr_pct):
    return min(price * atr_pct / 100.0, price * 0.08)


def test_forming_short_targets_stay_positive_on_huge_atr():
    price = 623.97
    atr = _capped(price, 25.0)            # KORU-like 25% ATR → capped to 8%
    stop, t1, t2 = forming_signals._levels("breakdown", price, atr, {})
    assert t1 > 0 and t2 > 0, f"targets went non-positive: t1={t1} t2={t2}"
    assert t2 < t1 < price                # short: targets below entry, ordered
    # without the cap, raw 25% ATR would have driven t2 negative
    assert (price - 5 * price * 0.25) < 0


def test_normal_atr_unaffected_by_cap():
    price = 100.0
    atr = _capped(price, 2.0)             # normal 2% ATR — well under the 8% cap
    assert abs(atr - 2.0) < 1e-9          # cap does not touch normal names
    stop, t1, t2 = forming_signals._levels("breakout", price, atr, {})
    assert t1 > price and t2 > t1         # long: targets above entry, ordered
