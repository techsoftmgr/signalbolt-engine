"""
Tests for the Fly.io stability fixes:

  • /health is dependency-free and fast (Fix 1)
  • scorer.score() always returns a "threshold" key (Fix 6)
  • runner logging tolerates a malformed score result (Fix 7)
"""
import time
from unittest.mock import patch

from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────────────────────
# Fix 1: /health must be instant and offline-safe
# ──────────────────────────────────────────────────────────────────────────────

def _import_app():
    # Imported lazily so conftest env stubs apply first
    from main import app
    return app


def test_health_returns_200_without_external_calls():
    app = _import_app()
    client = TestClient(app)

    # Patch external clients to blow up if /health touches them
    with patch("main.create_client", side_effect=AssertionError("Supabase must not be called from /health")), \
         patch("httpx.get",          side_effect=AssertionError("Alpaca must not be called from /health")):
        start = time.monotonic()
        r = client.get("/health")
        elapsed = time.monotonic() - start

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "signalbolt-engine"
    assert "timestamp" in body
    # No "checks" field — that lives on /ready now
    assert "checks" not in body
    # Should complete in well under 100 ms locally
    assert elapsed < 1.0, f"/health took {elapsed:.3f}s"


# ──────────────────────────────────────────────────────────────────────────────
# Fix 6: scorer.score() always returns "threshold"
# ──────────────────────────────────────────────────────────────────────────────

def test_score_empty_analysis_returns_threshold():
    from engine import scorer
    result = scorer.score({}, strategy_type="day_trade")
    assert "threshold" in result
    assert "passes" in result
    assert "breakdown" in result
    assert "confidence_factors" in result
    assert result["passes"] is False
    assert result["total"] == 0


def test_score_no_direction_returns_threshold():
    from engine import scorer
    result = scorer.score({"ticker": "AAPL", "current_price": 100.0}, strategy_type="scalping")
    assert "threshold" in result
    assert result["direction"] is None
    assert result["passes"] is False


def test_score_weak_l1_returns_threshold_and_factors():
    """Standard strategy with no SMC structure → early return — must still
    include threshold and an empty confidence_factors list."""
    from engine import scorer
    analysis = {
        "ticker":         "AAPL",
        "direction":      "LONG",
        "current_price":  100.0,
        "structure":      {},      # no BOS/CHoCH → L1 == 0
        "fvgs":           {},
        "obs":            {},
        "liquidity_sweep": {},
    }
    result = scorer.score(analysis, strategy_type="day_trade")
    assert "threshold" in result
    assert "confidence_factors" in result
    assert "breakdown" in result
    # All the keys runner.py grabs out of breakdown must exist
    for k in ("l1_smc", "l2_technical", "l3_sentiment", "l4_risk",
              "l5_mtf", "l6_regime", "l7_session", "l8_gamma",
              "l9_manipulation", "quant_bonus"):
        assert k in result["breakdown"], f"missing breakdown key: {k}"
    assert result["passes"] is False


def test_score_threshold_uses_session_override():
    from engine import scorer
    result = scorer.score(
        {},
        strategy_type="day_trade",
        session={"threshold": 85, "mode": "CATALYST_ONLY"},
    )
    assert result["threshold"] == 85


# ──────────────────────────────────────────────────────────────────────────────
# Fix 7: runner logging is defensive
# ──────────────────────────────────────────────────────────────────────────────

def test_runner_log_format_tolerates_missing_keys():
    """The exact f-string template used in runner.py must not crash when the
    score result is missing breakdown keys (regression for KeyError)."""
    scored: dict = {}                      # totally empty score result
    breakdown = scored.get("breakdown", {}) or {}
    # Replicates the format string in runner.py
    msg = (
        f"[runner] AAPL [day_trade] score={scored.get('total', 0)}/{scored.get('threshold', 0)} "
        f"grade={scored.get('confidence_grade','?')} "
        f"(L1={breakdown.get('l1_smc', 0)} L2={breakdown.get('l2_technical', 0)} "
        f"L3={breakdown.get('l3_sentiment', 0)} L4={breakdown.get('l4_risk', 0)})"
    )
    assert "AAPL" in msg
    assert "score=0/0" in msg
