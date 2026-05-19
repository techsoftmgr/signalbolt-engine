"""
Shared pytest fixtures for SignalBolt engine tests.

All external I/O is mocked here so tests:
  - Never hit real Supabase
  - Never call yfinance / Alpaca
  - Never send push notifications
  - Run in < 5 seconds total
"""
import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

# ── Add engine root to path ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Stub env vars before any engine import ───────────────────
os.environ.setdefault("SUPABASE_URL",        "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SECRET_KEY", "test_secret_key")
os.environ.setdefault("ALPACA_API_KEY",      "test_alpaca_key")
os.environ.setdefault("ALPACA_SECRET_KEY",   "test_alpaca_secret")
os.environ.setdefault("ALPACA_BASE_URL",     "https://paper-api.alpaca.markets")
os.environ.setdefault("ALPACA_DATA_FEED",    "iex")
os.environ.setdefault("ANTHROPIC_API_KEY",   "test_anthropic_key")
os.environ.setdefault("STRIPE_SECRET_KEY",   "sk_test_fake")
os.environ.setdefault("STRIPE_WEBHOOK_SECRET","whsec_fake")
os.environ.setdefault("STRIPE_PRO_PRICE_ID", "price_fake_pro")
os.environ.setdefault("STRIPE_PRO_PLUS_PRICE_ID", "price_fake_proplus")
os.environ.setdefault("SENTRY_DSN",          "")
os.environ.setdefault("ENVIRONMENT",         "test")
os.environ.setdefault("PORT",                "8000")

ET = ZoneInfo("America/New_York")


# ──────────────────────────────────────────────────────────────
# OHLCV fixtures
# ──────────────────────────────────────────────────────────────

def _make_ohlcv(
    n: int = 50,
    base_price: float = 150.0,
    trend: str = "up",   # "up" | "down" | "flat"
    volume: int = 500_000,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing."""
    import random
    random.seed(42)
    rows = []
    price = base_price
    for i in range(n):
        if trend == "up":
            price *= 1 + random.uniform(0, 0.005)
        elif trend == "down":
            price *= 1 - random.uniform(0, 0.005)
        else:
            price *= 1 + random.uniform(-0.002, 0.002)

        high  = price * 1.005
        low   = price * 0.995
        open_ = price * (1 + random.uniform(-0.002, 0.002))
        vol   = int(volume * random.uniform(0.8, 1.2))
        rows.append({"open": open_, "high": high, "low": low, "close": price, "volume": vol})

    return pd.DataFrame(rows)


@pytest.fixture
def ohlcv_uptrend() -> pd.DataFrame:
    return _make_ohlcv(50, 150.0, "up")


@pytest.fixture
def ohlcv_downtrend() -> pd.DataFrame:
    return _make_ohlcv(50, 150.0, "down")


@pytest.fixture
def ohlcv_flat() -> pd.DataFrame:
    return _make_ohlcv(50, 150.0, "flat")


@pytest.fixture
def ohlcv_short() -> pd.DataFrame:
    """DataFrame with too few rows — tests edge-case handling."""
    return _make_ohlcv(5, 150.0, "flat")


# ──────────────────────────────────────────────────────────────
# Supabase mock
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_supabase():
    """
    Full Supabase client mock. Chain: .table().select().eq().execute()
    returns MagicMock with .data = [].
    """
    sb = MagicMock()
    result = MagicMock()
    result.data = []
    # Support arbitrary chain depth
    sb.table.return_value.select.return_value.eq.return_value.execute.return_value = result
    sb.table.return_value.select.return_value.execute.return_value = result
    sb.table.return_value.insert.return_value.execute.return_value = result
    sb.table.return_value.update.return_value.eq.return_value.execute.return_value = result
    sb.table.return_value.upsert.return_value.execute.return_value = result
    return sb


@pytest.fixture
def mock_supabase_with_signals(mock_supabase):
    """Supabase mock pre-loaded with 3 active scalp signals."""
    signals = [
        {
            "id": "sig-001",
            "ticker": "AAPL",
            "direction": "LONG",
            "entry_price": 180.0,
            "stop_loss":   177.5,
            "target_one":  183.0,
            "target_two":  186.0,
            "strategy_type": "scalping",
            "status": "active",
            "result": "pending",
        },
        {
            "id": "sig-002",
            "ticker": "NVDA",
            "direction": "SHORT",
            "entry_price": 450.0,
            "stop_loss":   455.0,
            "target_one":  444.0,
            "target_two":  438.0,
            "strategy_type": "scalping",
            "status": "active",
            "result": "pending",
        },
        {
            "id": "sig-003",
            "ticker": "SPY",
            "direction": "LONG",
            "entry_price": 520.0,
            "stop_loss":   516.0,
            "target_one":  524.0,
            "target_two":  528.0,
            "strategy_type": "day_trade",
            "status": "active",
            "result": "pending",
        },
    ]
    result = MagicMock()
    result.data = signals
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = result
    return mock_supabase


# ──────────────────────────────────────────────────────────────
# yfinance mock
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_yfinance(ohlcv_uptrend):
    """Patch yfinance.Ticker so no real HTTP calls are made."""
    ticker_mock = MagicMock()
    ticker_mock.history.return_value = ohlcv_uptrend.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    ticker_mock.fast_info.last_price = 155.0
    ticker_mock.fast_info.previous_close = 150.0
    ticker_mock.info = {
        "trailingEps": 5.0,
        "earningsDate": None,
        "sector": "Technology",
    }
    ticker_mock.news = []

    with patch("yfinance.Ticker", return_value=ticker_mock):
        yield ticker_mock


# ──────────────────────────────────────────────────────────────
# Time helpers
# ──────────────────────────────────────────────────────────────

def et_time(hour: int, minute: int = 0, day: int = 14, month: int = 5, year: int = 2026) -> datetime:
    """Create a timezone-aware datetime in ET for a standard trading day (Wednesday)."""
    return datetime(year, month, day, hour, minute, tzinfo=ET)


@pytest.fixture
def standard_session_time():
    """10:30 AM ET — STANDARD session."""
    return et_time(10, 30)


@pytest.fixture
def orb_session_time():
    """9:50 AM ET — ORB session."""
    return et_time(9, 50)


@pytest.fixture
def catalyst_session_time():
    """9:35 AM ET — CATALYST_ONLY session."""
    return et_time(9, 35)


@pytest.fixture
def pre_market_time():
    """8:00 AM ET — PRE_MARKET."""
    return et_time(8, 0)


@pytest.fixture
def after_hours_time():
    """4:30 PM ET — AFTER_HOURS."""
    return et_time(16, 30)


@pytest.fixture
def fomc_time():
    """2:00 PM ET on a known FOMC date (2026-04-29)."""
    return et_time(14, 0, day=29, month=4, year=2026)


@pytest.fixture
def opex_time():
    """10:00 AM ET on OpEx day — 3rd Friday May 2026 = May 15."""
    return et_time(10, 0, day=15, month=5, year=2026)
