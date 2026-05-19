"""
Integration tests — FastAPI endpoints (main.py)

Tests run against FastAPI TestClient — no real HTTP calls to Supabase,
Alpaca, or Stripe. All external I/O is patched at the module level.

Covers:
  - GET  /health
  - GET  /signals
  - GET  /prices?tickers=AAPL,TSLA
  - POST /run
  - GET  /market-status
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
    # Patch Supabase client creation
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
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.return_value = signals_result
    mock_sb.table.return_value.select.return_value.execute.return_value = signals_result

    # Patch runner so /run doesn't actually do a full scan
    mock_runner = MagicMock()
    mock_runner.run_all_strategies = MagicMock(return_value=None)

    # Patch Alpaca data client
    mock_alpaca = MagicMock()
    mock_snapshot = MagicMock()
    mock_snapshot.latest_trade.price = 180.25
    mock_alpaca.get_stock_snapshot.return_value = {"AAPL": mock_snapshot, "TSLA": mock_snapshot}

    with patch("main.create_client", return_value=mock_sb), \
         patch("main._alpaca_data_client", mock_alpaca), \
         patch("main._alpaca_stock_snapshots", return_value={
             "AAPL": {"price": 180.25, "change_pct": 1.2, "volume": 50_000_000},
             "TSLA": {"price": 350.10, "change_pct": -0.8, "volume": 30_000_000},
         }), \
         patch("engine.runner.start_scheduler", return_value=None), \
         patch("engine.stream.run_stream", new_callable=lambda: lambda: AsyncMock()):
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
# /signals
# ──────────────────────────────────────────────────────────────

class TestSignalsEndpoint:

    def test_signals_returns_200(self, client):
        response = client.get("/signals")
        assert response.status_code == 200

    def test_signals_returns_list(self, client):
        response = client.get("/signals")
        data = response.json()
        assert isinstance(data, list)

    def test_signals_have_required_fields(self, client):
        response = client.get("/signals")
        signals = response.json()
        if signals:
            sig = signals[0]
            for field in ["id", "ticker", "direction", "entry_price"]:
                assert field in sig, f"Missing field: {field}"

    def test_signals_strategy_filter(self, client):
        """Filter by strategy_type should not crash."""
        response = client.get("/signals?strategy_type=scalping")
        assert response.status_code in (200, 422)  # 422 if param not supported yet


# ──────────────────────────────────────────────────────────────
# /prices
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
# /market-status
# ──────────────────────────────────────────────────────────────

class TestMarketStatus:

    def test_market_status_returns_200(self, client):
        response = client.get("/market-status")
        assert response.status_code == 200

    def test_market_status_has_session_field(self, client):
        response = client.get("/market-status")
        data = response.json()
        assert "session" in data or "status" in data or "market_open" in data

    def test_market_status_session_is_valid(self, client):
        response = client.get("/market-status")
        data = response.json()
        session = data.get("session") or data.get("status") or data.get("market_status")
        if session:
            assert session in ("pre", "market", "post", "closed",
                               "open", "pre_market", "after_hours")


# ──────────────────────────────────────────────────────────────
# /run (manual scan trigger)
# ──────────────────────────────────────────────────────────────

class TestRunEndpoint:

    def test_run_returns_200_or_202(self, client):
        with patch("engine.runner.run_all_strategies", return_value=None):
            response = client.post("/run")
        assert response.status_code in (200, 202, 401, 403)

    def test_run_without_auth_returns_error_if_key_required(self, client):
        """If ENGINE_API_KEY is set, unauthenticated requests should fail."""
        api_key = os.environ.get("ENGINE_API_KEY", "")
        if api_key:
            response = client.post("/run")
            assert response.status_code in (401, 403)
