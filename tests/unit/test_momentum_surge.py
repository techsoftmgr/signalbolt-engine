"""Momentum Surge detector — surge-window logic + disabled-mode guard."""
from engine import momentum_surge as ms


def test_surge_window():
    base = {"relativeVolume": 4.0, "price": 110, "ma20": 100}
    assert ms._is_surge({**base, "dayChangePct": 6.0}) is True       # young move on heavy vol, uptrend
    assert ms._is_surge({**base, "dayChangePct": 2.0}) is False      # move too small (< MIN_PCT)
    assert ms._is_surge({**base, "dayChangePct": 20.0}) is False     # already extended (> MAX_PCT) = chase
    assert ms._is_surge({**base, "dayChangePct": 6.0, "relativeVolume": 1.5}) is False  # not heavy enough
    assert ms._is_surge({**base, "dayChangePct": 6.0, "price": 95}) is False            # below MA = not uptrend
    assert ms._is_surge({"dayChangePct": 6.0}) is False              # missing fields → never fires


def test_run_disabled(monkeypatch):
    monkeypatch.setenv("MOMENTUM_SURGE_ENABLED", "false")
    assert ms._enabled() is False
    assert ms.run(sb=None) == {"scanned": 0, "candidates": 0, "fired": 0}
