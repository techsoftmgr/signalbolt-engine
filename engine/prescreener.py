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

import requests

logger = logging.getLogger("signalbolt.prescreener")

# ── Cache: re-screen every 2 min so new momentum movers are picked up quickly ─
_screen_cache: tuple[Optional[list[str]], float] = (None, 0.0)
_SCREEN_CACHE_TTL = 120   # 2 minutes

# ── Movers cache: top gainers + losers from Alpaca, refreshed every 5 min ────
# Separate from the snapshot cache so movers can be fetched independently.
_movers_cache: tuple[Optional[list[str]], float] = (None, 0.0)

# Minimum price for a dynamic mover to enter the scan universe. Below this,
# spreads are wide and stops slip badly (penny stocks / warrants).
_MIN_MOVER_PRICE = 5.0
_MOVERS_CACHE_TTL = 300   # 5 minutes

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


def fetch_movers(top: int = 30) -> list[str]:
    """
    Fetch today's top gainers and losers from Alpaca's screener endpoint.

    Returns up to 2×top tickers (30 gainers + 30 losers = 60 max).
    These are merged into EXTENDED_UNIVERSE before the snapshot screen runs,
    so stocks having their biggest day ever are never invisible to the engine
    just because they aren't on the fixed watchlist.

    Results are cached for 5 minutes — calling this every scan cycle is safe.
    Falls back to empty list if Alpaca is unavailable (fixed list still runs).
    """
    global _movers_cache
    cached, cached_at = _movers_cache
    if cached is not None and (time.monotonic() - cached_at) < _MOVERS_CACHE_TTL:
        return cached

    try:
        api_key    = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not api_secret:
            return []

        resp = requests.get(
            "https://data.alpaca.markets/v1beta1/screener/stocks/movers",
            params={"top": top},
            headers={
                "APCA-API-KEY-ID":     api_key,
                "APCA-API-SECRET-KEY": api_secret,
            },
            timeout=5,
        )

        if resp.status_code != 200:
            logger.warning(f"[screener] Movers endpoint returned {resp.status_code}")
            _movers_cache = ([], time.monotonic())
            return []

        data = resp.json()

        # Quality filter. The price check used to be only a COMMENT — penny
        # warrants like HUBCW ($0.05, 5 alpha chars) sailed through and fired a
        # swing signal that lost -31.8% in 9 min (spread alone is ~20% at $0.05).
        # Now actually enforce: real common shares only.
        def _ok(item: dict) -> bool:
            sym = (item.get("symbol") or "").upper()
            if not sym or not sym.isalpha() or len(sym) > 5:
                return False
            # Warrant / unit / rights: a 5-letter symbol ending W/U/R is almost
            # always a derivative (HUBCW, …), not a tradeable common share.
            if len(sym) == 5 and sym[-1] in ("W", "U", "R"):
                return False
            # Skip sub-$MIN penny names — thin, wide spreads, stops slip badly.
            price = item.get("price")
            try:
                if price is not None and float(price) < _MIN_MOVER_PRICE:
                    return False
            except (TypeError, ValueError):
                pass
            return True

        gainers = [item["symbol"] for item in data.get("gainers", []) if _ok(item)]
        losers  = [item["symbol"] for item in data.get("losers",  []) if _ok(item)]
        clean   = gainers + losers

        logger.info(
            f"[screener] Movers: {len(data.get('gainers',[]))} gainers + "
            f"{len(data.get('losers',[]))} losers → {len(clean)} after quality filter "
            f"(min ${_MIN_MOVER_PRICE:.0f}, no warrants/units)"
        )

        _movers_cache = (clean, time.monotonic())
        return clean

    except Exception as e:
        logger.warning(f"[screener] Movers fetch failed: {e} — using fixed universe only")
        _movers_cache = ([], time.monotonic())
        return []


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

        # ── Merge live movers into the fixed universe ─────────────────────────
        # fetch_movers() returns today's top 30 gainers + 30 losers from Alpaca.
        # Any mover not already in EXTENDED_UNIVERSE gets appended so stocks
        # having a breakout day are never invisible to the engine.
        movers       = fetch_movers(top=30)
        universe_set = set(EXTENDED_UNIVERSE)
        new_movers   = [t for t in movers if t not in universe_set]
        scan_universe = EXTENDED_UNIVERSE + new_movers

        if new_movers:
            logger.info(
                f"[screener] Added {len(new_movers)} live mover(s) not in fixed universe: "
                f"{new_movers[:10]}{'...' if len(new_movers) > 10 else ''}"
            )

        client = StockHistoricalDataClient(api_key, api_secret)

        # Single batch call — returns snapshot for all symbols at once.
        # Alpaca paid SIP plan handles this easily with no rate-limit issues.
        req = StockSnapshotRequest(symbol_or_symbols=scan_universe)
        snapshots = client.get_stock_snapshot(req)

        # Score every ticker: combine move strength + volume ratio into one score
        # so the final list is sorted by "how interesting is this ticker right now"
        scored: list[tuple[str, float]] = []

        # Time-of-day volume projection factor (computed ONCE for the whole batch).
        # Alpaca's snapshot daily_bar.volume is today's PARTIAL volume intraday, so
        # comparing it raw to yesterday's FULL volume makes vol_ratio artificially
        # low early (~0.14x at 9:46am) and MISSES genuinely high-volume names. Scale
        # today's volume up to a full-day estimate via the shared empirical intraday
        # curve before the ratio. Only during RTH (pre/post-market: use raw).
        from datetime import datetime as _dt, timezone as _tz
        from engine.volume_curve import expected_volume_fraction
        _elapsed  = _dt.now(_tz.utc).hour * 60 + _dt.now(_tz.utc).minute - (13 * 60 + 30)
        _vol_proj = (1.0 / max(0.05, expected_volume_fraction(_elapsed))) if 0 < _elapsed < 390 else 1.0

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

                vol_ratio = (today_vol * _vol_proj) / prev_vol if prev_vol > 0 else 1.0

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
            f"[screener] {len(scan_universe)} tickers scanned "
            f"({len(EXTENDED_UNIVERSE)} fixed + {len(new_movers)} live movers) → "
            f"{len(scored)} qualify "
            f"(move≥{min_move_pct*100:.1f}% OR gap≥{min_gap_pct*100:.1f}% OR vol≥{min_vol_ratio}×) → "
            f"{len(result)} sent to SMC (cap={max_results})"
        )

        # ── Subscribe new movers to the live WebSocket stream ────────────────
        # Any ticker that just appeared from the movers endpoint needs to be
        # added to the Alpaca trade stream so it gets tick-by-tick level
        # monitoring immediately — not just when a signal is fired on it.
        if new_movers:
            try:
                import asyncio
                from engine import stream as _stream
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(
                        _stream.subscribe_extra_tickers(new_movers)
                    )
                else:
                    loop.run_until_complete(
                        _stream.subscribe_extra_tickers(new_movers)
                    )
            except Exception as _sub_err:
                logger.debug(f"[screener] Mover WebSocket subscribe skipped: {_sub_err}")

        _screen_cache = (result, time.monotonic())
        return result

    except Exception as e:
        logger.warning(f"[screener] Alpaca snapshot failed: {e} — falling back to core tickers")
        fallback = list(CORE_TICKERS)
        _screen_cache = (fallback, time.monotonic())
        return fallback


def invalidate_cache() -> None:
    """Force a fresh screen on the next call (e.g. after market open)."""
    global _screen_cache
    _screen_cache = (None, 0.0)
