"""
Unit tests — engine/phantom_audit.py

Verifies the EOD data-integrity audit flags PHANTOM (recorded exit the tape never
printed), MISMATCH, NO_PNL, and OVERSHOOT — and leaves clean closes alone.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import pytest

from engine import phantom_audit as pa


# ── Fake Supabase query chain ───────────────────────────────────────────────
class _FakeQuery:
    def __init__(self, rows): self._rows = rows
    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def gte(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def execute(self): return SimpleNamespace(data=self._rows)


class _FakeSB:
    def __init__(self, rows): self._rows = rows
    def table(self, _name): return _FakeQuery(self._rows)


def _bars(low, high):
    idx = pd.to_datetime(
        ["2026-06-03T15:00:00Z", "2026-06-03T17:55:00Z", "2026-06-03T18:30:00Z"], utc=True)
    return pd.DataFrame({"high": [high, high, high], "low": [low, low, low],
                         "close": [(low + high) / 2] * 3}, index=idx)


def _sig(**kw):
    base = dict(id="x", ticker="CMCSA", direction="SHORT", entry_price=23.7,
                stop_loss=24.62, result="loss", result_pct=-3.9,
                closed_reason="stop_hit", closed_at="2026-06-03T17:55:00Z",
                score_breakdown={"detector_source": "BREAKDOWN"})
    base.update(kw)
    return base


def _run(rows, low, high):
    with patch("engine.alpaca_client.get_bars", return_value=_bars(low, high)):
        return pa.audit(sb=_FakeSB(rows), days=1)


class TestPhantomAudit:
    def test_clean_close_not_flagged(self):
        # SHORT stop at 24.62, exit ~24.62 within day range [23.5, 24.9] → clean
        res = _run([_sig(result_pct=-3.9)], low=23.5, high=24.9)
        assert res["serious_count"] == 0
        assert res["audited"] == 1

    def test_phantom_exit_outside_range_flagged(self):
        # recorded -11.8% → exit ~26.5, but day high only 24.9 → PHANTOM
        res = _run([_sig(result="loss", result_pct=-11.82)], low=23.5, high=24.9)
        kinds = {f["kind"] for f in res["flagged"]}
        assert "PHANTOM" in kinds
        assert res["serious_count"] >= 1

    def test_result_pct_mismatch_flagged(self):
        # result=win but result_pct negative → MISMATCH
        res = _run([_sig(result="win", result_pct=-3.9)], low=23.5, high=24.9)
        kinds = {f["kind"] for f in res["flagged"]}
        assert "MISMATCH" in kinds
        assert res["serious_count"] >= 1

    def test_no_pnl_on_stop_flagged(self):
        res = _run([_sig(result="loss", result_pct=None, closed_reason="stop_hit")],
                   low=23.5, high=24.9)
        kinds = {f["kind"] for f in res["flagged"]}
        assert "NO_PNL" in kinds

    def test_overshoot_is_warn_not_serious(self):
        # loss -7% but stop distance ~3.9%; exit 22.04 (LONG-style) — use within range
        # SHORT entry 23.7, -7% → exit ~25.36; put day high above it so it's in-range
        res = _run([_sig(result="loss", result_pct=-7.0)], low=23.5, high=25.6)
        kinds = {f["kind"] for f in res["flagged"]}
        assert "OVERSHOOT" in kinds
        assert res["serious_count"] == 0   # overshoot alone is not serious

    def test_unverified_when_no_bars(self):
        with patch("engine.alpaca_client.get_bars", return_value=None):
            res = pa.audit(sb=_FakeSB([_sig()]), days=1)
        assert res["unverified"] == 1
        assert res["serious_count"] == 0   # can't verify → don't false-flag
