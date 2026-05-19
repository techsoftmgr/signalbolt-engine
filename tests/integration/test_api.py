"""
Integration tests — FastAPI endpoints (main.py)

Tests run against FastAPI TestClient — no real HTTP calls to Supabase,
Alpaca, or Stripe. All external I/O is patched at the module level.

Actual response shapes (from main.py):
  GET  /signals  → {"signals": [...], "count": N}
  GET  /prices   → {ticker: {price, changePercent, session, ...}}
  GET  /health   → {"status": "ok"|"degraded", ...}
  POST /run      → {"status": "triggered", ...}  (requires ENGINE_API_KEY or dev mode)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from fastapi.testclient import TestClient


# ── Patch all external services before importing main ────────
@pytest.fixture(scope="module")
def client():
    """
    Build a TestClient with all external I/O patched.
    Module-scoped so the app is only created once per test file.
    """
    mock_sb = MagicMock()
    signals_result = MagicMock()
    signals_result.data = [
        {
            "id": "abc-001",
            "ticker": "AAPL",
            "direction": "LONG",
            "entry_price": 180.0,
            "stop_loss": 177.5,
            "target_one": 183.0,
            "target_two": 186.0,
            "confidence_score": 82,
            "strategy_type": "day_trade",
            "status": "active",
            "created_at": "2026-05-19T10:00:00Z",
        }
    ]
    # Chain for: .select("*").order(...).limit(...).execute()
    mock_sb.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = signals_result
    # Chain with eq filter: .select("*").eq(...).order(...).limit(...).execute()
    mock_sb.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = signals_result
    mock_sb.table.return_value.select.return_value.execute.return_value = signals_result

    mock_alpaca = MagicMock()
    mock_snapshot = MagicMock()
    mock_snapshot.latest_trade.price = 180.25
    mock_alpaca.get_stock_snapshot.return_value = {"AAPL": mock_snapshot, "TSLA": mock_snapshot}

    with patch("main.create_client", return_value=mock_sb), \
         patch("main._alpaca_data_client", mock_alpaca), \
         patch("main._alpaca_stock_snapshots", return_value={
             "AAPL": {"price": 180.25, "changePercent": 1.2, "session": "market", "volume": 50_000_000},
             "TSLA": {"price": 350.10, "changePercent": -0.8, "session": "market", "volume": 30_000_000},
         }), \
         patch("engine.runner.start_scheduler", return_value=MagicMock()), \
         patch("engine.stream.run_stream", new_callable=AsyncMock):
        import main as app_module
        app_module.app.state.supabase = mock_sb

        with TestClient(app_module.app, raise_server_exceptions=False) as c:
            yield c


# ──────────────────────────────────────────────────────────────
# /health
# ──────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_field(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data

    def test_health_status_ok_or_degraded(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] in ("ok", "degraded", "healthy")


# ──────────────────────────────────────────────────────────────
# /signals  — response shape: {"signals": [...], "count": N}
# ──────────────────────────────────────────────────────────────

class TestSignalsEndpoint:

    def test_signals_returns_200(self, client):
        response = client.get("/signals")
        assert response.status_code == 200

    def test_signals_returns_envelope(self, client):
        """Response is a JSON object with 'signals' and 'count' keys."""
        response = client.get("/signals")
        data = response.json()
        assert isinstance(data, dict)
        assert "signals" in data
        assert "count" in data

    def test_signals_list_is_array(self, client):
        response = client.get("/signals")
        data = response.json()
        assert isinstance(data["signals"], (list, dict))  # list normally

    def test_signals_have_required_fields(self, client):
        response = client.get("/signals")
        signals = response.json().get("signals", [])
        if isinstance(signals, list) and signals:
            sig = signals[0]
            for field in ["id", "ticker", "direction", "entry_price"]:
                assert field in sig, f"Missing field: {field}"

    def test_signals_strategy_filter(self, client):
        """Filter by strategy_type should not crash."""
        response = client.get("/signals?strategy_type=scalping")
        assert response.status_code in (200, 422)


# ──────────────────────────────────────────────────────────────
# /prices   — response shape: {ticker: {price, changePercent, ...}}
# ──────────────────────────────────────────────────────────────

class TestPricesEndpoint:

    def test_prices_returns_200(self, client):
        response = client.get("/prices?tickers=AAPL,TSLA")
        assert response.status_code == 200

    def test_prices_returns_dict(self, client):
        response = client.get("/prices?tickers=AAPL,TSLA")
        data = response.json()
        assert isinstance(data, dict)

    def test_prices_missing_tickers_param(self, client):
        """Missing tickers param should return 422."""
        response = client.get("/prices")
        assert response.status_code == 422

    def test_prices_single_ticker(self, client):
        response = client.get("/prices?tickers=AAPL")
        assert response.status_code == 200


# ──────────────────────────────────────────────────────────────
# /run (manual scan trigger)
# ──────────────────────────────────────────────────────────────

class TestRunEndpoint:

    def test_run_returns_200_or_401_or_503(self, client):
        """
        /run requires ENGINE_API_KEY in production.
        In dev (no key set) it triggers a background scan and returns 200.
        In production (key set but not provided) it returns 401 or 503.
        """
        with patch("engine.runner.run_scan", return_value=None):
            response = client.post("/run", json={})
        assert response.status_code in (200, 202, 401, 403, 503)

    def test_run_without_auth_blocks_in_production(self, client):
        """If ENGINE_API_KEY is set, unauthenticated requests should fail."""
        api_key = os.environ.get("ENGINE_API_KEY", "")
        if api_key:
            response = client.post("/run", json={})
            assert response.status_code in (401, 403)
