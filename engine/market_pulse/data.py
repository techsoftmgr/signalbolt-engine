"""
Market Pulse — data fetch. Index + constituent OHLCV come from Alpaca (reusing the
shared client). VIX comes from a SECONDARY source (yfinance ^VIX, Cboe fallback)
and is fully ISOLATED: any VIX failure returns None and never touches pillars 1-4.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("signalbolt.market_pulse.data")

_CHUNK = 120   # symbols per Alpaca multi-bar request


def index_bars(symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
    """Daily OHLCV for one index ETF (SPY/QQQ)."""
    try:
        from engine.alpaca_client import get_bars
        return get_bars(symbol, timeframe="1Day", days=days)
    except Exception as e:
        logger.warning(f"[market_pulse] index_bars({symbol}) failed: {e}")
        return None


def universe_bars(tickers: list[str], days: int = 400) -> dict[str, pd.DataFrame]:
    """Daily OHLCV for the whole constituent list (~1 trading year), fetched in
    chunks. Missing names are simply omitted — breadth is a % of what we got."""
    out: dict[str, pd.DataFrame] = {}
    if not tickers:
        return out
    try:
        from engine.alpaca_client import get_multi_bars
    except Exception as e:
        logger.warning(f"[market_pulse] alpaca import failed: {e}")
        return out
    for i in range(0, len(tickers), _CHUNK):
        chunk = tickers[i:i + _CHUNK]
        try:
            out.update(get_multi_bars(chunk, "1Day", days) or {})
        except Exception as e:
            logger.warning(f"[market_pulse] universe chunk {i} failed: {e}")
    return out


# ── VIX (secondary source — ISOLATED) ───────────────────────────────────────
def vix_closes(lookback: int = 40) -> Optional[pd.Series]:
    """Chronological series of recent VIX closes from a NON-Alpaca source. Returns
    None on any failure (caller then computes the regime from pillars 1-4 only).

    Primary: yfinance ^VIX. Fallback: Stooq CSV (Cboe-derived). Each in its own
    guard so neither can break the feature."""
    # 1) yfinance ^VIX
    try:
        import yfinance as yf
        df = yf.Ticker("^VIX").history(period=f"{max(lookback, 20)}d", interval="1d")
        if df is not None and not df.empty and "Close" in df:
            s = df["Close"].dropna()
            if len(s):
                return s.astype(float).tail(lookback)
    except Exception as e:
        logger.warning(f"[market_pulse] yfinance ^VIX failed: {e}")

    # 2) Stooq daily history (free CSV, Cboe-derived) — fallback only
    try:
        import io
        import pandas as _pd
        import requests
        r = requests.get("https://stooq.com/q/d/l/?s=^vix&i=d", timeout=10)
        if r.ok and r.text and "Date" in r.text[:50]:
            df = _pd.read_csv(io.StringIO(r.text))
            if "Close" in df and len(df):
                return df["Close"].dropna().astype(float).tail(lookback)
    except Exception as e:
        logger.warning(f"[market_pulse] stooq ^VIX fallback failed: {e}")

    logger.warning("[market_pulse] VIX unavailable from all sources — regime will use pillars 1-4 only")
    return None
