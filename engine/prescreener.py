"""
Pre-screener — fast first-pass filter before full SMC analysis.

Scans a large universe of 500+ tickers in ~15 seconds using Alpaca's
/v2/stocks/snapshots endpoint (single batch call, no per-ticker loops).
Returns the top N tickers that show momentum + volume conditions worth
running full SMC on.

Criteria (any one qualifies):
  1. Price moved ≥ 0.8% in last bar (momentum)
  2. Volume ≥ 1.5× 30-day average (unusual activity)
  3. Ticker is in the CORE watchlist (always included regardless)

This reduces the SMC workload from 500+ tickers to ~30-50 per scan
while covering the full liquid US market for signal opportunities.
"""

import logging
import os
import time
from typing import Optional

logger = logging.getLogger("signalbolt.prescreener")

# ── Cache: avoid re-screening within 5 min ───────────────────────────────────
_screen_cache: tuple[Optional[list[str]], float] = (None, 0.0)
_SCREEN_CACHE_TTL = 300   # 5 minutes

# ── Always include these regardless of pre-screen results ────────────────────
# Core liquid tickers that generate the most reliable SMC setups.
CORE_TICKERS = [
    "SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT", "AMD", "META",
    "GOOGL", "AMZN", "IWM", "COIN", "PLTR",
]

# ── Extended watchlist — 500 liquid US stocks ─────────────────────────────────
# Sourced from S&P 500 + Russell 1000 + high-volume growth names.
# Pre-screener picks the ~30-50 showing momentum or volume from this list.
EXTENDED_UNIVERSE = [
    # ── Mega-cap tech ──────────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN", "TSLA",
    "NFLX", "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "AMAT", "LRCX",
    "KLAC", "MRVL", "SMCI", "ARM", "ON", "NXPI", "STX", "WDC",
    # ── Software / cloud ───────────────────────────────────────
    "CRM", "NOW", "SNOW", "PLTR", "DDOG", "NET", "CRWD", "ZS", "OKTA",
    "FTNT", "PANW", "TEAM", "SHOP", "MELI", "SE", "GRAB", "U", "RBLX",
    "COIN", "HOOD", "SOFI", "AFRM", "UPST", "LC",
    # ── Indices / ETFs ─────────────────────────────────────────
    "SPY", "QQQ", "IWM", "DIA", "TQQQ", "SQQQ", "UVXY", "VXX",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLRE",
    "GLD", "SLV", "TLT", "HYG", "LQD", "ARKK", "ARKG", "ARKW",
    # ── Financials ─────────────────────────────────────────────
    "JPM", "GS", "MS", "BAC", "C", "WFC", "BLK", "SCHW", "CME",
    "V", "MA", "PYPL", "SQ", "AXP", "COF", "DFS",
    # ── Health / biotech ───────────────────────────────────────
    "MRNA", "BNTX", "PFE", "JNJ", "LLY", "ABBV", "BMY", "GILD",
    "BIIB", "REGN", "VRTX", "SGEN", "EXAS", "RARE", "ACAD",
    # ── Energy ─────────────────────────────────────────────────
    "XOM", "CVX", "COP", "SLB", "HAL", "OXY", "DVN", "MPC", "VLO",
    "PSX", "LNG", "AR", "EQT", "FANG", "PXD",
    # ── Consumer / retail ──────────────────────────────────────
    "AMZN", "WMT", "TGT", "COST", "HD", "LOW", "NKE", "LULU",
    "SBUX", "MCD", "CMG", "DPZ", "YUM", "EL", "PG", "KO", "PEP",
    # ── Autos / EV ─────────────────────────────────────────────
    "TSLA", "GM", "F", "RIVN", "LCID", "NIO", "LI", "XPEV",
    # ── Real estate / REITs ────────────────────────────────────
    "AMT", "PLD", "EQIX", "DLR", "SPG", "O", "VICI",
    # ── Industrials ────────────────────────────────────────────
    "GE", "HON", "MMM", "CAT", "DE", "UPS", "FDX", "LMT", "RTX",
    "BA", "NOC", "GD",
    # ── Media / telecom ────────────────────────────────────────
    "DIS", "CMCSA", "T", "VZ", "TMUS", "PARA", "WBD", "NFLX",
    # ── Crypto-adjacent ────────────────────────────────────────
    "COIN", "MSTR", "MARA", "RIOT", "CLSK", "CIFR", "HUT", "BTBT",
    # ── Small/mid-cap momentum ─────────────────────────────────
    "MSTR", "PLTR", "HOOD", "DKNG", "PENN", "SFIX", "RKT", "UWMC",
    "OPEN", "CVNA", "CARVANA", "BYND", "OUST", "LIDR",
    # ── Travel / leisure ───────────────────────────────────────
    "UBER", "LYFT", "ABNB", "BKNG", "EXPE", "MAR", "HLT", "CCL",
    "RCL", "NCLH", "DAL", "UAL", "AAL", "LUV",
    # ── Healthcare / insurance ─────────────────────────────────
    "UNH", "CVS", "CI", "HUM", "CNC", "MOH", "MCK", "ABC", "CAH",
]

# Deduplicate while preserving order
_seen: set[str] = set()
_deduped: list[str] = []
for _t in EXTENDED_UNIVERSE:
    if _t not in _seen:
        _seen.add(_t)
        _deduped.append(_t)
EXTENDED_UNIVERSE = _deduped


def screen(
    max_results: int = 150,
    min_move_pct: float = 0.008,   # 0.8% intraday move qualifies
    min_vol_ratio: float = 1.5,    # 1.5× average volume qualifies
) -> list[str]:
    """
    Return tickers showing momentum or unusual volume today (up to max_results).

    The pre-screener covers the full EXTENDED_UNIVERSE in ONE Alpaca batch
    snapshot call (~3 seconds). Full SMC only runs on tickers that pass.
    max_results=150 is safe on shared-cpu-1x Fly.io — completes in ~5 min,
    well within the 15-min scan cycle.

    On a quiet market day, fewer tickers qualify so the list is naturally
    shorter. On high-volatility days more pass, up to the cap.

    Falls back to CORE_TICKERS if Alpaca is unavailable.
    """
    global _screen_cache

    # Return cached result within TTL
    cached, cached_at = _screen_cache
    if cached is not None and (time.monotonic() - cached_at) < _SCREEN_CACHE_TTL:
        logger.debug(f"[screener] Cache hit — {len(cached)} tickers")
        return cached

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockSnapshotRequest

        api_key    = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not api_secret:
            raise ValueError("Alpaca keys not set")

        client = StockHistoricalDataClient(api_key, api_secret)

        # Single batch call — returns snapshot for all ~300 symbols at once.
        # Alpaca paid SIP plan handles this easily with no rate-limit issues.
        req = StockSnapshotRequest(symbol_or_symbols=EXTENDED_UNIVERSE)
        snapshots = client.get_stock_snapshot(req)

        # Score every ticker: combine move strength + volume ratio into one score
        # so the final list is sorted by "how interesting is this ticker right now"
        scored: list[tuple[str, float]] = []

        for ticker, snap in snapshots.items():
            try:
                daily = snap.daily_bar
                prev  = snap.previous_daily_bar

                if not daily or not prev:
                    continue

                open_price  = float(daily.open)
                close_price = float(daily.close)
                today_vol   = float(daily.volume)
                prev_vol    = float(prev.volume)

                if open_price <= 0 or prev_vol <= 0:
                    continue

                move_pct  = abs(close_price - open_price) / open_price
                vol_ratio = today_vol / prev_vol if prev_vol > 0 else 1.0

                # Only include tickers above EITHER threshold
                if move_pct < min_move_pct and vol_ratio < min_vol_ratio:
                    continue

                # Combined score: weighted sum of normalised move + vol
                # Move contributes 60%, volume 40% — momentum is the primary signal
                interest_score = (move_pct / min_move_pct) * 0.6 + (vol_ratio / min_vol_ratio) * 0.4
                scored.append((ticker, interest_score))

            except Exception:
                continue

        # Sort by interest score (highest first) — best setups get SMC first
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build result: always include CORE tickers, then add qualifiers
        result: list[str] = list(CORE_TICKERS)
        seen = set(result)

        for ticker, score in scored:
            if len(result) >= max_results:
                break
            if ticker not in seen:
                result.append(ticker)
                seen.add(ticker)

        logger.info(
            f"[screener] {len(EXTENDED_UNIVERSE)} tickers scanned → "
            f"{len(scored)} qualify (move≥{min_move_pct*100:.1f}% or vol≥{min_vol_ratio}×) → "
            f"{len(result)} sent to SMC (cap={max_results})"
        )

        _screen_cache = (result, time.monotonic())
        return result

    except Exception as e:
        logger.warning(f"[screener] Alpaca snapshot failed: {e} — using core tickers")
        fallback = list(CORE_TICKERS)
        _screen_cache = (fallback, time.monotonic())
        return fallback


def invalidate_cache() -> None:
    """Force a fresh screen on the next call (e.g. after market open)."""
    global _screen_cache
    _screen_cache = (None, 0.0)
