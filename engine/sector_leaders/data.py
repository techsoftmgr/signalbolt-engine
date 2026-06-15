"""Sector Leaders — data fetch (Alpaca daily bars for the 11 ETFs + SPY)."""
from __future__ import annotations

import logging

import pandas as pd

from . import config as C

logger = logging.getLogger("signalbolt.sector_leaders.data")


def fetch_bars(days: int = 400) -> dict[str, pd.DataFrame]:
    """{symbol: daily OHLCV} for the 11 sector ETFs + SPY. {} on failure."""
    try:
        from engine.alpaca_client import get_multi_bars
        return get_multi_bars(C.ETFS + [C.BENCHMARK], "1Day", days) or {}
    except Exception as e:
        logger.warning(f"[sector_leaders] fetch_bars failed: {e}")
        return {}
