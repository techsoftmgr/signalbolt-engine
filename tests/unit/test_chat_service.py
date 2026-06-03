"""
Unit tests — engine/chat_service.py

Covers history cleaning, grounded-context building, and graceful no-API-key behavior.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from types import SimpleNamespace
from unittest.mock import patch
import pytest

from engine import chat_service as cs


class _FakeQuery:
    def __init__(self, rows): self._rows = rows
    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def execute(self): return SimpleNamespace(data=self._rows)


class _FakeSB:
    def __init__(self, rows): self._rows = rows
    def table(self, _n): return _FakeQuery(self._rows)


class TestCleanHistory:
    def test_filters_invalid_and_truncates(self):
        hist = [{"role": "system", "content": "x"}] + \
               [{"role": "user", "content": f"m{i}"} for i in range(10)]
        out = cs._clean_history(hist)
        assert len(out) == cs._MAX_HISTORY            # capped
        assert all(m["role"] in ("user", "assistant") for m in out)
        assert out[-1]["content"] == "m9"             # keeps the LAST turns

    def test_drops_empty(self):
        out = cs._clean_history([{"role": "user", "content": "   "}])
        assert out == []

    def test_none_safe(self):
        assert cs._clean_history(None) == []


class TestBuildContext:
    def test_includes_quant_expert_and_signals(self):
        sigs = [{"direction": "SHORT", "entry_price": 100, "stop_loss": 103,
                 "target_one": 97, "target_two": 94, "strategy_type": "breakdown",
                 "confidence_score": 70}]
        cr = {"taBias": "bearish", "short_term": "weak", "agreement": "agree",
              "idea": "SHORT idea", "narrative": "broke down", "patterns": []}
        with patch("engine.quant_score_service.cached_score",
                   return_value=({"trendScore": -40, "setupType": "breakdown",
                                  "ma20": 105, "atrPct": 2.0, "dayChangePct": -3}, None)), \
             patch("engine.chart_read.analyze", return_value=cr):
            ctx = cs._build_context(_FakeSB(sigs), "NVDA")
        assert "TICKER: NVDA" in ctx
        assert "QUANT:" in ctx and "breakdown" in ctx
        assert "EXPERT_READ:" in ctx and "agree" in ctx
        assert "TRADE_IDEA:" in ctx
        assert "ACTIVE_SIGNAL: SHORT breakdown entry=100" in ctx

    def test_no_signals_notes_none(self):
        with patch("engine.quant_score_service.cached_score", return_value=(None, None)), \
             patch("engine.chart_read.analyze", return_value=None):
            ctx = cs._build_context(_FakeSB([]), "AAPL")
        assert "ACTIVE_SIGNAL: none right now" in ctx


class TestAnswer:
    def test_no_api_key_is_graceful(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            res = cs.answer(_FakeSB([]), "NVDA", "what's the entry?", [])
        assert res["ok"] is False
        assert isinstance(res["answer"], str) and res["answer"]

    def test_empty_message_rejected(self):
        res = cs.answer(_FakeSB([]), "NVDA", "   ", [])
        assert res["ok"] is False
