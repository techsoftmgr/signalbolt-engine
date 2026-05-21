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
    "TWLO", "BILL", "ZM", "ROKU", "TTD", "GTLB", "MNDY", "DBX",
    # ── High-growth / momentum names ──────────────────────────
    # These are frequently the biggest intraday/earnings movers — must be in universe
    "SPOT", "RDDT", "APP", "CELH", "DUOL", "ONON", "DECK", "ELF",
    "HIMS", "SNAP", "PINS", "DOCS", "WOLF", "TTWO", "EA", "RIVN",
    "HOOD", "AFRM", "UPST", "OPEN", "COUR", "BMBL", "JOBY",
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
    "OPEN", "CVNA", "BYND", "OUST", "LIDR",
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
    min_gap_pct:  float = 0.015,   # 1.5% overnight gap qualifies (earnings, news)
    min_vol_ratio: float = 1.5,    # 1.5× average volume qualifies
) -> list[str]:
    """
    Return tickers showing momentum, overnight gap, or unusual volume today.

    THREE ways to qualify (any one is enough):
      1. Intraday move ≥ 0.8%    (open → close of today's bar)
      2. Overnight gap  ≥ 1.5%   (prev close → today's open) — catches EARNINGS
      3. Volume ≥ 1.5× prior day

    WHY gap detection matters:
      An earnings gap stock opens at a new level and often CONSOLIDATES intraday.
      open→close move may be only 0.2% but the overnight gap is +12%.
      Without gap detection these stocks are invisible to the screener — exactly
      why ARM (gap up) and WMT (gap down) were missed on earnings days.

    The pre-screener covers the full EXTENDED_UNIVERSE in ONE Alpaca batch
    snapshot call (~3 seconds). Full SMC only runs on tickers that pass.
    max_results=150 is safe on shared-cpu-1x Fly.io — completes in ~5 min,
    well within the 15-min scan cycle.

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
                prev_close  = float(prev.close)
                prev_vol    = float(prev.volume)

                if open_price <= 0 or prev_vol <= 0 or prev_close <= 0:
                    continue

                # ── Intraday move: open → close of today's bar ──────────
                move_pct = abs(close_price - open_price) / open_price

                # ── Overnight gap: prev close → today's open ────────────
                # This is the key fix: earnings gaps open at a new level and
                # may consolidate intraday (tiny open→close move) but the
                # overnight gap is the actual signal.  Both ARM (+gap up) and
                # WMT (-gap down) show here while the intraday move is small.
                gap_pct = abs(open_price - prev_close) / prev_close

                vol_ratio = today_vol / prev_vol if prev_vol > 0 else 1.0

                # Qualify on ANY of: intraday move, overnight gap, or volume
                has_move = move_pct >= min_move_pct
                has_gap  = gap_pct  >= min_gap_pct
                has_vol  = vol_ratio >= min_vol_ratio
                if not has_move and not has_gap and not has_vol:
                    continue

                # Use the stronger of the two move signals for scoring.
                # Gap stocks get an extra 0.5× bonus so they bubble to the
                # front of the list — fresh gap setups are higher-priority.
                best_move = max(move_pct, gap_pct)
                gap_bonus = 0.5 if has_gap else 0.0

                # Combined score: move 55%, volume 35%, gap bonus 10%
                interest_score = (
                    (best_move / min_move_pct) * 0.55
                    + (vol_ratio / min_vol_ratio) * 0.35
                    + gap_bonus
                )
                if has_gap and not has_move:
                    logger.debug(
                        f"[screener] {ticker} gap-qualified — "
                        f"gap={gap_pct*100:.1f}% intraday={move_pct*100:.1f}%"
                    )
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
            f"{len(scored)} qualify "
            f"(move≥{min_move_pct*100:.1f}% OR gap≥{min_gap_pct*100:.1f}% OR vol≥{min_vol_ratio}×) → "
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
