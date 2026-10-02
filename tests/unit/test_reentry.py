"""Shakeout re-entry (engine.reentry) — pins the re-engagement rules so the asymmetric
capture fix (+43 pts measured) can't silently regress. Default-OFF; only re-enters a
LOSING momentum shakeout that RECLAIMS its entry, holds the 20-EMA, and still re-scores
in the original direction — within the reclaim window, deduped, capped."""
import types
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from engine import reentry


# ── fakes ─────────────────────────────────────────────────────────────────────
class _Q:
    def __init__(self, data): self._data = data
    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def gte(self, *a, **k): return self
    def contains(self, *a, **k): self._data = []; return self   # dup-check → none
    def limit(self, *a, **k): return self
    def execute(self): return types.SimpleNamespace(data=self._data)


class _SB:
    def __init__(self, rows): self.rows = rows
    def table(self, *_): return _Q(list(self.rows))


def _daily(n=220, last=110.0, entry=100.0, closed_days_ago=3):
    """n daily bars trending up to `last`; closed_at is `closed_days_ago` bars back."""
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize().tz_localize(None), periods=n, freq="D")
    closes = np.linspace(entry * 0.9, last, n)
    return pd.DataFrame({"open": closes, "high": closes + 1, "low": closes - 1,
                         "close": closes, "volume": [1e6] * n}, index=idx)


def _ms(bias="LONG"):
    return types.SimpleNamespace(ticker="MU", bias=bias, score=2.0, last_price=110.0,
                                 atr=3.0, raw_return=0.2, ann_vol=0.3, sma_fast=105.0,
                                 sma_slow=95.0, ext_atr=0.5)


def _sig(**over):
    s = {"id": "orig1", "ticker": "MU", "direction": "LONG", "entry_price": 100.0,
         "status": "closed", "result_pct": -4.0, "closed_reason": "stop_hit",
         "closed_at": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
         "origin": None, "score_breakdown": {"detector_source": "TREND_MOMENTUM"}}
    s.update(over); return s


def _wire(monkeypatch, *, bars=None, bias="LONG", active=False):
    from engine import alpaca_client, momentum_detector as md, runner
    fired = []
    monkeypatch.setattr(alpaca_client, "get_bars", lambda *a, **k: bars if bars is not None else _daily())
    monkeypatch.setattr(md, "score", lambda *a, **k: _ms(bias))
    monkeypatch.setattr(runner, "_has_active_signal", lambda *a, **k: active)
    monkeypatch.setattr(runner, "_fire_momentum",
                        lambda sb, ms, direction, reentry=None: fired.append((ms.ticker, direction, reentry)))
    return fired


# ── kill switch ────────────────────────────────────────────────────────────────
def test_default_off(monkeypatch):
    monkeypatch.delenv("RE_ENTRY_ENABLED", raising=False)
    assert reentry.enabled() is False
    fired = _wire(monkeypatch)
    reentry.scan(_SB([_sig()]))
    assert fired == []


# ── happy path ───────────────────────────────────────────────────────────────
def test_reenters_a_reclaimed_loser(monkeypatch):
    monkeypatch.setenv("RE_ENTRY_ENABLED", "true")
    fired = _wire(monkeypatch)                          # bars trend up to 110 > entry 100 > EMA
    out = reentry.scan(_SB([_sig()]))
    assert out["reentered"] == 1
    tk, direction, tag = fired[0]
    assert tk == "MU" and direction == "LONG"
    assert tag["origin"] == "orig1" and tag["attempt"] == 1 and tag["original_entry"] == 100.0


# ── only losing shakeouts ──────────────────────────────────────────────────────
def test_skips_winners(monkeypatch):
    monkeypatch.setenv("RE_ENTRY_ENABLED", "true")
    fired = _wire(monkeypatch)
    reentry.scan(_SB([_sig(result_pct=12.0)]))           # a winner — nothing to re-enter
    assert fired == []


# ── reclaim required ───────────────────────────────────────────────────────────
def test_no_reentry_without_reclaim(monkeypatch):
    monkeypatch.setenv("RE_ENTRY_ENABLED", "true")
    # price never gets back above the 100 entry (stays ~85-95)
    fired = _wire(monkeypatch, bars=_daily(last=95.0))
    reentry.scan(_SB([_sig()]))
    assert fired == []


# ── window ─────────────────────────────────────────────────────────────────────
def test_skips_when_past_reclaim_window(monkeypatch):
    monkeypatch.setenv("RE_ENTRY_ENABLED", "true")
    monkeypatch.setenv("RE_ENTRY_RECLAIM_DAYS", "5")
    fired = _wire(monkeypatch)
    reentry.scan(_SB([_sig(closed_at=(datetime.now(timezone.utc) - timedelta(days=20)).isoformat())]))
    assert fired == []                                   # reclaim came too late


# ── re-score must still agree ──────────────────────────────────────────────────
def test_skips_when_trend_gone(monkeypatch):
    monkeypatch.setenv("RE_ENTRY_ENABLED", "true")
    fired = _wire(monkeypatch, bias="NONE")              # reclaimed but momentum re-score != LONG
    reentry.scan(_SB([_sig()]))
    assert fired == []


# ── dedup ──────────────────────────────────────────────────────────────────────
def test_dedup_when_already_active(monkeypatch):
    monkeypatch.setenv("RE_ENTRY_ENABLED", "true")
    fired = _wire(monkeypatch, active=True)
    reentry.scan(_SB([_sig()]))
    assert fired == []


# ── attempt cap ────────────────────────────────────────────────────────────────
def test_respects_max_reentries(monkeypatch):
    monkeypatch.setenv("RE_ENTRY_ENABLED", "true")
    monkeypatch.setenv("RE_ENTRY_MAX", "1")
    fired = _wire(monkeypatch)
    already = _sig(score_breakdown={"detector_source": "TREND_MOMENTUM",
                                    "reentry": {"origin": "orig1", "attempt": 1}})
    reentry.scan(_SB([already]))
    assert fired == []                                   # already re-entered once
