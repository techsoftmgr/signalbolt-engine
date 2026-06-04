"""
Heatmap Service — computes per-ticker market metrics for the heatmap tab.

Data flow:
  Alpaca batch daily bars  → previous close, avg volume
  Alpaca batch 5-min bars  → intraday momentum, VWAP approximation
  Alpaca batch latest prices → real-time current price

Cache TTL: 15 seconds (HEATMAP_CACHE_TTL env var).
All scores degrade gracefully — missing Alpaca data returns None fields
rather than crashing.
"""

import logging
import time
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("signalbolt.heatmap")

# Empirical intraday CUMULATIVE volume curve — fraction of a full RTH day's volume
# typically completed by N minutes after the 9:30 ET open. Derived from ~340
# ticker-days of 5-min bars across the universe (2026-06). Volume is heavily
# front-loaded by the opening surge (~14% in the first 15 min), so the old naive
# `elapsed / 390` assumption massively OVER-projected early-session relative volume
# — e.g. HOOD 2026-06-04 9:46am: real ~0.8x opening volume was projected to a fake
# "2.3x" and fired a false accumulation signal. Projecting against this curve
# instead makes relative volume valid at ANY time of day.
_VOL_CURVE = [
    (0, 0.0), (5, 0.087), (10, 0.113), (15, 0.139), (20, 0.160), (30, 0.200),
    (45, 0.255), (60, 0.306), (90, 0.387), (120, 0.459), (180, 0.570),
    (240, 0.670), (300, 0.768), (360, 0.880), (390, 1.0),
]


def _expected_volume_fraction(elapsed_min: float) -> float:
    """Fraction of a full RTH day's volume typically done by `elapsed_min` minutes
    after the open (linear-interpolated empirical curve). Clamped to (0, 1]."""
    if elapsed_min >= 390:
        return 1.0
    if elapsed_min <= 5:
        return _VOL_CURVE[1][1]          # floor at the 5-min mark (~8.7%); avoids
                                          # div-by-zero + tames the first-bar noise
    for (m0, f0), (m1, f1) in zip(_VOL_CURVE, _VOL_CURVE[1:]):
        if m0 <= elapsed_min <= m1:
            t = (elapsed_min - m0) / (m1 - m0) if m1 > m0 else 0.0
            return f0 + t * (f1 - f0)
    return 1.0

# ── In-memory cache ──────────────────────────────────────────────────────────
_cache: list[dict]   = []
_cache_ts: float     = 0.0
_CACHE_TTL: int      = int(os.environ.get("HEATMAP_CACHE_TTL", "15"))

# Default Pro-tier ticker universe (mirrors runner.py watchlist)
DEFAULT_TICKERS: list[str] = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMD",
    "COIN", "PLTR", "MSTR", "HOOD", "RBLX", "UBER", "ABNB",
    "JPM", "GS", "XOM", "CVX",
    "MARA", "RIOT", "CLSK", "MRNA", "BNTX",
]

# Sector mapping for sector filter
TICKER_SECTORS: dict[str, str] = {
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "GOOGL": "Tech",
    "META": "Tech", "AMD": "Tech",
    "TSLA": "EV/Auto", "RBLX": "Gaming", "UBER": "Transport", "ABNB": "Travel",
    "COIN": "Crypto", "MSTR": "Crypto", "MARA": "Crypto", "RIOT": "Crypto", "CLSK": "Crypto",
    "PLTR": "AI/Defense", "HOOD": "Fintech",
    "JPM": "Finance", "GS": "Finance",
    "XOM": "Energy", "CVX": "Energy",
    "MRNA": "Biotech", "BNTX": "Biotech",
}

# Market cap cache (USD). Hand-curated baseline so the treemap always has
# size data even when yfinance is slow / rate-limited. Refreshed daily from
# yfinance in _refresh_market_caps(); these numbers are May 2026 approximations
# and used as fallback only.
_BASELINE_MARKET_CAPS: dict[str, float] = {
    # Mega-caps (trillions)
    "AAPL":  3_400_000_000_000,  "MSFT":  3_100_000_000_000,
    "NVDA":  3_000_000_000_000,  "GOOGL": 2_100_000_000_000,
    "META":  1_400_000_000_000,
    # Large-caps (hundreds of billions)
    "TSLA":  900_000_000_000,    "AMD":   270_000_000_000,
    "JPM":   600_000_000_000,    "XOM":   500_000_000_000,
    "CVX":   300_000_000_000,    "GS":    140_000_000_000,
    # Mid-caps (tens of billions)
    "UBER":   140_000_000_000,   "ABNB":   90_000_000_000,
    "COIN":   80_000_000_000,    "MSTR":   90_000_000_000,
    "PLTR":   150_000_000_000,   "HOOD":   30_000_000_000,
    "RBLX":   30_000_000_000,    "MRNA":   30_000_000_000,
    "BNTX":   25_000_000_000,
    # Small-caps (single-digit billions)
    "MARA":   6_000_000_000,     "RIOT":   3_500_000_000,
    "CLSK":   3_000_000_000,
    # ETFs — use AUM as proxy
    "SPY":   600_000_000_000,    "QQQ":   300_000_000_000,
    "IWM":    65_000_000_000,    "DIA":    35_000_000_000,
}

# Daily-refreshing live cap cache (overrides baseline when fresh)
_market_cap_cache: dict[str, float] = {}
_market_cap_ts: float = 0.0
_MARKET_CAP_TTL: int = int(os.environ.get("MARKET_CAP_TTL", "86400"))  # 24h


def _get_market_cap(ticker: str) -> float:
    """
    Return market cap in USD. Tries the live yfinance-backed cache first,
    falls back to the baseline. Never raises.
    """
    live = _market_cap_cache.get(ticker)
    if live and live > 0:
        return live
    return _BASELINE_MARKET_CAPS.get(ticker, 1_000_000_000)  # 1B floor for unknowns


def _refresh_market_caps_if_stale(tickers: list[str]) -> None:
    """
    Once a day, refresh market caps from yfinance for the watchlist. Runs
    inline on first cache miss after TTL — fast enough (~3-5s for 27 tickers)
    that it's not worth a background job. Failures are silent; baselines
    cover us.
    """
    global _market_cap_ts
    now = time.monotonic()
    if now - _market_cap_ts < _MARKET_CAP_TTL and _market_cap_cache:
        return

    try:
        import yfinance as yf
        # Use yf.Tickers (plural) for a batched info fetch
        joined = " ".join(tickers)
        for t in tickers:
            try:
                info = yf.Ticker(t).fast_info
                cap = float(getattr(info, "market_cap", None) or 0)
                if cap > 0:
                    _market_cap_cache[t] = cap
            except Exception:
                continue
        _market_cap_ts = now
        logger.info(f"[heatmap] Refreshed market caps for {len(_market_cap_cache)} tickers")
    except Exception as e:
        logger.warning(f"[heatmap] Market cap refresh failed (using baselines): {e}")


def _safe_float(val, default: float = 0.0) -> float:
    """Convert to float safely, returning default on any error."""
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def compute_heatmap(
    symbols: Optional[list[str]] = None,
    active_signals: Optional[dict[str, str]] = None,
    sort_by: str = "momentum",
    filter_by: Optional[str] = None,
    sector: Optional[str] = None,
    min_rel_volume: float = 0.0,
) -> list[dict]:
    """
    Main heatmap entry point.

    Args:
        symbols:        tickers to include (None = use default universe)
        active_signals: {ticker: signal_id} map from Supabase active signals
        sort_by:        "momentum" | "gainers" | "losers" | "volume"
        filter_by:      "bullish" | "bearish" | "high_volume" | None
        sector:         sector name filter, or None
        min_rel_volume: minimum relative volume threshold

    Returns list of ticker dicts, sorted and filtered per params.
    """
    global _cache, _cache_ts

    now = time.monotonic()
    if now - _cache_ts < _CACHE_TTL and _cache:
        raw = _cache
    else:
        raw = _build_fresh(symbols or DEFAULT_TICKERS, active_signals or {})
        if raw:
            _cache    = raw
            _cache_ts = now

    # Apply filters
    result = raw[:]
    if filter_by == "bullish":
        result = [r for r in result if r["momentumScore"] > 20]
    elif filter_by == "bearish":
        result = [r for r in result if r["momentumScore"] < -20]
    elif filter_by == "high_volume":
        result = [r for r in result if r["relativeVolume"] >= 1.5]

    if sector:
        result = [r for r in result if r.get("sector") == sector]
    if min_rel_volume > 0:
        result = [r for r in result if r["relativeVolume"] >= min_rel_volume]

    # Sort
    if sort_by == "gainers":
        result.sort(key=lambda x: x["dayChangePct"], reverse=True)
    elif sort_by == "losers":
        result.sort(key=lambda x: x["dayChangePct"])
    elif sort_by == "volume":
        result.sort(key=lambda x: x["relativeVolume"], reverse=True)
    else:  # default: momentum
        result.sort(key=lambda x: x["momentumScore"], reverse=True)

    return result


def _build_fresh(tickers: list[str], active_signals: dict[str, str]) -> list[dict]:
    """Fetch data from Alpaca and compute all metrics. Called on cache miss."""
    from engine.alpaca_client import get_latest_prices, get_multi_bars

    # Refresh market caps once a day in-band. Cheap on cache hit.
    _refresh_market_caps_if_stale(tickers)

    try:
        # Three batch calls instead of N×3 individual calls
        daily_bars    = get_multi_bars(tickers, timeframe="1Day", days=22)
        intraday_bars = get_multi_bars(tickers, timeframe="5Min", days=2)
        latest_prices = get_latest_prices(tickers)

        results: list[dict] = []
        for ticker in tickers:
            try:
                row = _compute_ticker(
                    ticker,
                    latest_prices.get(ticker),
                    daily_bars.get(ticker),
                    intraday_bars.get(ticker),
                    active_signals,
                )
                if row:
                    results.append(row)
            except Exception as e:
                logger.debug(f"[heatmap] {ticker}: {e}")

        return results

    except Exception as e:
        logger.error(f"[heatmap] _build_fresh failed: {e}")
        return []


def _compute_ticker(
    ticker: str,
    latest_price: Optional[float],
    daily_df,
    intraday_df,
    active_signals: dict[str, str],
) -> Optional[dict]:
    """Compute full metric set for one ticker. Returns None if data is insufficient."""
    import pandas as pd
    from datetime import datetime, timezone

    # ── Current price ─────────────────────────────────────────────────────────
    current_price = latest_price
    if current_price is None and daily_df is not None and not daily_df.empty:
        current_price = _safe_float(daily_df["close"].iloc[-1])
    if current_price is None:
        return None

    # ── Previous close & average volume ──────────────────────────────────────
    prev_close = None
    avg_volume = 0.0
    today_open = None

    if daily_df is not None and len(daily_df) >= 2:
        prev_close  = _safe_float(daily_df["close"].iloc[-2])
        # 20-day average volume (excluding today)
        avg_volume  = _safe_float(daily_df["volume"].iloc[:-1].tail(20).mean())
        today_open  = _safe_float(daily_df["open"].iloc[-1])

    if prev_close is None or prev_close == 0:
        return None

    day_change_pct = round((current_price - prev_close) / prev_close * 100, 2)

    # ── Intraday metrics ──────────────────────────────────────────────────────
    price_5m_ago  : Optional[float] = None
    price_15m_ago : Optional[float] = None
    vwap          : Optional[float] = None
    current_volume: float = 0.0

    if intraday_df is not None and not intraday_df.empty:
        today = datetime.now(timezone.utc).date()

        # Filter to today's bars only
        try:
            today_mask = intraday_df.index.date == today
            today_intra = intraday_df[today_mask]
        except Exception:
            today_intra = intraday_df.tail(78)

        if not today_intra.empty:
            current_volume = _safe_float(today_intra["volume"].sum())

            if len(today_intra) >= 2:
                price_5m_ago = _safe_float(today_intra["close"].iloc[-2])
            if len(today_intra) >= 4:
                price_15m_ago = _safe_float(today_intra["close"].iloc[-4])

            # VWAP = Σ(typical_price × volume) / Σ(volume)
            vol = today_intra["volume"]
            typ = (today_intra["high"] + today_intra["low"] + today_intra["close"]) / 3
            total_vol = _safe_float(vol.sum())
            if total_vol > 0:
                vwap = _safe_float((typ * vol).sum() / total_vol)

    # ── Relative volume ───────────────────────────────────────────────────────
    rel_volume = 1.0
    if avg_volume > 0 and current_volume > 0:
        now_utc = datetime.now(timezone.utc)
        # Market 9:30–16:00 ET = 13:30–20:00 UTC. Project today's volume-so-far to a
        # full day using the EMPIRICAL intraday volume curve (front-loaded), NOT a
        # naive elapsed/390 — see _VOL_CURVE. This is the proper "relative volume at
        # this time of day" and is valid at the open as well as midday.
        market_open_utc_mins = 13 * 60 + 30
        elapsed = now_utc.hour * 60 + now_utc.minute - market_open_utc_mins
        exp_frac = _expected_volume_fraction(elapsed)
        projected = current_volume / max(exp_frac, 0.05)
        rel_volume = round(projected / avg_volume, 2)

    # ── Momentum score ────────────────────────────────────────────────────────
    momentum_score = _momentum_score(
        current_price, prev_close, price_5m_ago, price_15m_ago, vwap, rel_volume,
    )

    # ── Volatility score (10-day std dev of daily returns × 100) ─────────────
    vol_score = 0.0
    if daily_df is not None and len(daily_df) >= 5:
        ret = daily_df["close"].pct_change().tail(10).dropna()
        if len(ret) >= 3:
            vol_score = round(_safe_float(ret.std()) * 100, 2)

    # ── Trend direction ───────────────────────────────────────────────────────
    if momentum_score >= 40:
        trend = "bullish"
    elif momentum_score <= -40:
        trend = "bearish"
    else:
        trend = "neutral"

    return {
        "ticker":          ticker,
        "sector":          TICKER_SECTORS.get(ticker, "Other"),
        "price":           round(current_price, 2),
        "prevClose":       round(prev_close, 2),
        "dayChangePct":    day_change_pct,
        "relativeVolume":  rel_volume,
        "vwap":            round(vwap, 2) if vwap else None,
        "trendDirection":  trend,
        "momentumScore":   momentum_score,
        "volatilityScore": vol_score,
        # colorIntensity: 0-100 for green/red tile shading
        "colorIntensity":  min(100, int(abs(day_change_pct) * 15)),
        # marketCap: USD. Drives treemap rectangle size in the app.
        "marketCap":       _get_market_cap(ticker),
        "hasSignal":       ticker in active_signals,
        "signalId":        active_signals.get(ticker),
    }


def _momentum_score(
    price: float,
    prev_close: float,
    price_5m: Optional[float],
    price_15m: Optional[float],
    vwap: Optional[float],
    rel_volume: float,
) -> int:
    """
    Momentum score: -100 to +100.
    Bullish > 60 | Bearish < -60 | Neutral -40 to +40.

    Components:
      35% — day change vs previous close
      20% — 5-min price change
      20% — 15-min price change
      15% — price vs VWAP
      10% — volume spike direction amplifier
    """
    score = 0.0

    # Day change component
    day_chg = (price - prev_close) / prev_close * 100
    score  += float(np.clip(day_chg * 5, -35, 35))

    # 5-min momentum
    if price_5m and price_5m > 0:
        chg_5m = (price - price_5m) / price_5m * 100
        score += float(np.clip(chg_5m * 10, -20, 20))

    # 15-min momentum
    if price_15m and price_15m > 0:
        chg_15m = (price - price_15m) / price_15m * 100
        score  += float(np.clip(chg_15m * 7, -20, 20))

    # VWAP relationship
    if vwap and vwap > 0:
        vwap_pct = (price - vwap) / vwap * 100
        score   += float(np.clip(vwap_pct * 5, -15, 15))

    # Volume spike — amplifies direction
    if rel_volume > 1.5:
        direction = 1 if day_chg >= 0 else -1
        score    += direction * min(10, (rel_volume - 1) * 4)

    return int(np.clip(round(score), -100, 100))
