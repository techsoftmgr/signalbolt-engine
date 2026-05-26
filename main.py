import json
import os
import sys
import time
import logging
import requests

# Force UTF-8 stdout/stderr so emoji in log messages don't crash on Windows
# (cp1252 default encoding can't encode characters like ✅ ⚡ ⏱).
# This is a no-op on Linux/Railway (already UTF-8).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from typing import List, Optional

import sentry_sdk
import stripe
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client, acreate_client, AsyncClient

from engine.config import (
    ANTHROPIC_API_KEY,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    STRIPE_SECRET_KEY,
    STRIPE_WEBHOOK_SECRET,
    STRIPE_PRO_PRICE_ID,
    STRIPE_PRO_PLUS_PRICE_ID,
    SENTRY_DSN,
    ENVIRONMENT,
    ENGINE_PUBLIC_URL,
    ENGINE_API_KEY,
)

load_dotenv()

# Sentry error monitoring
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        environment=ENVIRONMENT,
        traces_sample_rate=0.1,
        profiles_sample_rate=0.1,
    )

stripe.api_key = STRIPE_SECRET_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("signalbolt")

POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "")

# ── Alpaca data client singleton ──────────────────────────────
# Fix #5: _alpaca_stock_snapshots() was creating a new StockHistoricalDataClient
# on every /prices request. At 100 subscribers refreshing every 30s that's
# 200 new HTTP clients per minute. One module-level instance is shared.
_alpaca_data_client = None
try:
    from alpaca.data.historical import StockHistoricalDataClient as _AlpacaDataClient
    from alpaca.data.requests import StockSnapshotRequest as _AlpacaSnapshotReq
    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        _alpaca_data_client = _AlpacaDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
except Exception as _e:
    logger.debug(f"[main] Alpaca data client init failed: {_e}")

# ── /indices response cache ───────────────────────────────────
# App polls every 5 s. Without a cache each poll = Alpaca + yfinance calls.
# Cache stores the last computed result and its timestamp; result is reused
# if it is less than 5 seconds old. Alpaca/yfinance are only hit once per
# 5-second window regardless of how many users are connected.
_indices_cache: dict = {}
_indices_cache_ts: float = 0.0
# 15s — app polls /indices on tab focus + every 15s. A 3s cache fired 5x per
# poll cycle; 15s aligns with one Alpaca/Polygon call per cycle. Cuts Alpaca
# spend ~5x at 1k users from ~600/min to ~120/min on this endpoint alone.
_INDICES_CACHE_TTL: float = 15.0

# Per-ticker price cache so the app can poll every 5 s without hammering Alpaca.
# Keys are individual ticker symbols; each entry is (timestamp, price_dict).
_prices_cache: dict[str, tuple[float, dict]] = {}
# 15s — REST /prices is a fallback for tickers not on the live WS stream.
# The WS delivers tick-by-tick anyway, so the REST path doesn't need
# sub-second freshness. 15s matches the app poll cycle = 1 Alpaca call
# per tab refresh instead of 3-5.
_PRICES_CACHE_TTL: float = 15.0


def _market_session() -> str:
    """Return 'pre', 'market', 'post', or 'closed' based on US Eastern time."""
    now = datetime.now(ZoneInfo('America/New_York'))
    if now.weekday() >= 5:          # Weekend
        return 'closed'
    t = now.hour * 60 + now.minute
    if t < 4 * 60:       return 'closed'
    if t < 9 * 60 + 30:  return 'pre'
    if t < 16 * 60:      return 'market'
    if t < 20 * 60:      return 'post'
    return 'closed'


def _supabase_key() -> str:
    return os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]


# Singleton Supabase client — created once on first use, reused across all
# requests. Previously _make_supabase() built a fresh client per call, which
# means a TLS handshake on every endpoint hit. Under load that's the difference
# between 50ms and 500ms per request.
_sb_singleton: Optional[Client] = None


def _make_supabase() -> Client:
    global _sb_singleton
    if _sb_singleton is None:
        _sb_singleton = create_client(os.environ["SUPABASE_URL"], _supabase_key())
    return _sb_singleton


# ── Async Supabase client (singleton) ─────────────────────────────────────────
# For endpoints converted to async — they get true non-blocking I/O instead of
# tying up an anyio threadpool thread for each Supabase call. Migration is
# gradual; sync endpoints keep using _make_supabase() above.
#
# Usage:
#   sb = await _make_supabase_async()
#   rows = (await sb.table("foo").select("*").execute()).data
_sb_async_singleton: Optional[AsyncClient] = None
_sb_async_lock = None  # asyncio.Lock — created lazily inside event loop


async def _make_supabase_async() -> AsyncClient:
    global _sb_async_singleton, _sb_async_lock
    if _sb_async_singleton is not None:
        return _sb_async_singleton
    import asyncio
    if _sb_async_lock is None:
        _sb_async_lock = asyncio.Lock()
    async with _sb_async_lock:
        if _sb_async_singleton is None:
            _sb_async_singleton = await acreate_client(
                os.environ["SUPABASE_URL"], _supabase_key()
            )
    return _sb_async_singleton


# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------

def _polygon_stock_snapshots(symbols: list[str]) -> dict:
    """Bulk stock snapshot from Polygon — price + day change + extended hours for multiple tickers."""
    if not POLYGON_KEY or not symbols:
        return {}
    try:
        r = requests.get(
            "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers",
            params={"tickers": ",".join(symbols), "apiKey": POLYGON_KEY},
            timeout=8,
        )
        session = _market_session()
        result  = {}
        for t in r.json().get("tickers", []):
            sym        = t["ticker"]
            day_price  = float(t.get("day",     {}).get("c") or 0)
            prev_close = float(t.get("prevDay", {}).get("c") or 0)
            reg_price  = day_price or prev_close
            if reg_price <= 0:
                continue

            reg_chg = float(
                t.get("todaysChangePerc")
                or (((reg_price - prev_close) / prev_close * 100) if prev_close else 0)
            )

            entry: dict = {
                "price":         round(reg_price, 2),
                "changePercent": round(reg_chg, 2),
                "session":       session,
            }

            if session == "pre":
                ext = float(t.get("preMarket", {}).get("c") or 0)
                if ext > 0 and prev_close > 0:
                    entry["extendedPrice"]         = round(ext, 2)
                    entry["extendedChangePercent"] = round((ext - prev_close) / prev_close * 100, 2)
            elif session == "post":
                ext = float(t.get("afterHours", {}).get("c") or 0)
                if ext > 0 and reg_price > 0:
                    entry["extendedPrice"]         = round(ext, 2)
                    entry["extendedChangePercent"] = round((ext - reg_price) / reg_price * 100, 2)

            result[sym] = entry
        logger.debug(f"[polygon] snapshots ({session}) for {list(result.keys())}")
        return result
    except Exception as e:
        logger.debug(f"[polygon] stock snapshots error: {e}")
        return {}


def _alpaca_btc_snapshot() -> Optional[dict]:
    """
    Real-time BTC/USD from Alpaca crypto feed — available 24/7, free on all plans.
    Alpaca crypto uses 'BTC/USD' format (not 'BTC-USD' like yfinance).
    """
    if not ALPACA_API_KEY:
        return None
    try:
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoSnapshotRequest

        # Crypto client needs no auth for market data (public feed)
        client   = CryptoHistoricalDataClient()
        snapshot = client.get_crypto_snapshot(
            CryptoSnapshotRequest(symbol_or_symbols=["BTC/USD"])
        )
        snap = snapshot.get("BTC/USD")
        if not snap:
            return None

        price     = float(snap.latest_trade.price) if snap.latest_trade else 0.0
        prev_bar  = snap.previous_daily_bar
        prev_close = float(prev_bar.close) if prev_bar else price
        chg       = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

        if price > 0:
            logger.debug(f"[alpaca] BTC/USD = ${price:,.2f} ({chg:+.2f}%)")
            return {"price": round(price, 2), "changePercent": round(chg, 2)}
    except Exception as e:
        logger.debug(f"[alpaca] BTC snapshot error: {e}")
    return None


def _polygon_crypto_snapshot(symbol: str) -> Optional[dict]:
    """Single crypto snapshot from Polygon — fallback if Alpaca fails."""
    if not POLYGON_KEY:
        return None
    try:
        r = requests.get(
            f"https://api.polygon.io/v2/snapshot/locale/global/markets/crypto/tickers/{symbol}",
            params={"apiKey": POLYGON_KEY},
            timeout=8,
        )
        t = r.json().get("ticker", {})
        price = float(t.get("day", {}).get("c") or t.get("prevDay", {}).get("c") or 0)
        prev  = float(t.get("prevDay", {}).get("c") or price)
        chg   = float(t.get("todaysChangePerc") or (((price - prev) / prev * 100) if prev else 0))
        if price > 0:
            return {"price": round(price, 2), "changePercent": round(chg, 2)}
    except Exception as e:
        logger.debug(f"[polygon] crypto snapshot error: {e}")
    return None


def _yf_price(ticker: str) -> Optional[dict]:
    """yfinance fallback for a single ticker."""
    try:
        import yfinance as yf
        info  = yf.Ticker(ticker).fast_info
        price = float(info.last_price or info.previous_close or 0)
        prev  = float(info.previous_close or price)
        chg   = ((price - prev) / prev * 100) if prev else 0.0
        if price > 0:
            return {
                "price":         round(price, 2),
                "changePercent": round(chg, 2),
                "session":       _market_session(),
            }
    except Exception as e:
        logger.debug(f"[yfinance] {ticker}: {e}")
    return None


# ---------------------------------------------------------------------------
# Alpaca snapshot (primary price source)
# ---------------------------------------------------------------------------

def _alpaca_stock_snapshots(symbols: list[str]) -> dict:
    """
    Bulk real-time snapshot from Alpaca SIP feed.
    Returns price + changePercent + extended hours for all symbols in one call.

    During market hours:   latest_trade.price = real-time last trade
    During pre-market:     minute_bar.close   = latest pre-market price
    During post-market:    minute_bar.close   = latest after-hours price

    Falls back gracefully to {} on any error.
    """
    if not ALPACA_API_KEY or not symbols or _alpaca_data_client is None:
        return {}
    try:
        from alpaca.data.requests import StockSnapshotRequest

        snapshots = _alpaca_data_client.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=symbols)
        )

        session = _market_session()
        result  = {}

        for sym, snap in snapshots.items():
            try:
                # ── Regular session price ─────────────────────────────
                trade_price = float(snap.latest_trade.price) if snap.latest_trade else 0.0
                prev_close  = float(snap.previous_daily_bar.close) if snap.previous_daily_bar else 0.0
                day_close   = float(snap.daily_bar.close) if snap.daily_bar else trade_price

                # Use latest trade during market; daily bar close outside hours
                reg_price = trade_price if session == "market" else (day_close or trade_price)

                if reg_price <= 0:
                    continue

                chg_pct = ((reg_price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

                entry: dict = {
                    "price":         round(reg_price, 2),
                    "changePercent": round(chg_pct, 2),
                    "session":       session,
                }

                # ── Extended hours price (pre / post market) ──────────
                # Alpaca minute_bar gives the most recent 1-min bar
                # which covers pre-market (4AM) and after-hours (to 8PM)
                if session in ("pre", "post") and snap.minute_bar:
                    ext_price = float(snap.minute_bar.close)
                    if ext_price > 0:
                        entry["extendedPrice"] = round(ext_price, 2)
                        if session == "pre" and prev_close > 0:
                            entry["extendedChangePercent"] = round(
                                (ext_price - prev_close) / prev_close * 100, 2
                            )
                        elif session == "post" and day_close > 0:
                            entry["extendedChangePercent"] = round(
                                (ext_price - day_close) / day_close * 100, 2
                            )

                result[sym] = entry

            except Exception as e:
                logger.debug(f"[alpaca] snapshot parse error for {sym}: {e}")
                continue

        logger.debug(f"[alpaca] snapshots ({session}) for {list(result.keys())}")
        return result

    except Exception as e:
        logger.debug(f"[alpaca] snapshots error: {e}")
        return {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

# ── Process-level toggle ──────────────────────────────────────────────────────
# The trading engine (scheduler + Alpaca WebSocket + price broadcast) is heavy
# and was previously started inside the web process. That single VM ended up
# doing API + scanning + WebSocket + push, which is what triggered Fly's
# PR04 load-balancer warnings and made /health time out.
#
# Default: web process is API-only. Set RUN_ENGINE_IN_WEB=true to revert to the
# old behaviour. The worker process (engine/worker.py) is where the engine runs
# in production.
RUN_ENGINE_IN_WEB = os.getenv("RUN_ENGINE_IN_WEB", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    scheduler         = None
    stream_task       = None
    broadcast_task    = None

    if not RUN_ENGINE_IN_WEB:
        logger.info("SignalBolt API started — engine disabled in web process (RUN_ENGINE_IN_WEB=false)")
        yield
        logger.info("SignalBolt API stopped")
        return

    from engine.runner import start_scheduler
    from engine.stream import run_stream
    from engine import price_store

    # ── Initialise real-time price store with this event loop ────────────────
    price_store.init(asyncio.get_running_loop())

    # ── Seed price store from Alpaca REST snapshot so first WS connect gets
    #    immediate data even before the first trade arrives ───────────────────
    try:
        from engine.runner import ALL_TICKERS
        seed_tickers = list(dict.fromkeys(ALL_TICKERS))[:40]
        snaps = _alpaca_stock_snapshots(seed_tickers)
        for ticker, data in snaps.items():
            price_store.seed(ticker, data["price"], data["changePercent"], data.get("session", "market"))
            chg = data["changePercent"]
            prev = data["price"] / (1 + chg / 100) if chg != -100 else data["price"]
            price_store.set_prev_close(ticker, prev)
        logger.info(f"[lifespan] Price store seeded with {len(snaps)} tickers")
    except Exception as e:
        logger.warning(f"[lifespan] Price store seed failed (non-fatal): {e}")

    # ── APScheduler: day_trade / swing / options_flow / dark_pool / maintenance ──
    scheduler = start_scheduler()

    # ── Alpaca WebSocket: bars (signal scanning) + trades (price broadcast) ──
    stream_task = asyncio.create_task(run_stream(), name="alpaca_stream")

    # ── 10 Hz price broadcast loop ────────────────────────────────────────────
    async def _price_broadcast_loop():
        while True:
            await asyncio.sleep(0.1)   # 100 ms = 10 Hz
            try:
                await price_store.broadcast_snapshot()
            except Exception as _be:
                logger.debug(f"[price_broadcast] error: {_be}")

    broadcast_task = asyncio.create_task(
        _price_broadcast_loop(), name="price_broadcast"
    )

    logger.info(
        "SignalBolt engine started in WEB process — "
        "scalping=WebSocket real-time | day_trade/swing=APScheduler | "
        "prices=10 Hz WebSocket push to app"
    )
    yield

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    try:
        from engine.stream import _wss_ref as _stream_wss
        if _stream_wss is not None:
            _stream_wss.stop()
            logger.info("[lifespan] Alpaca WebSocket stopped — connection released")
            await asyncio.sleep(2)
    except Exception as _se:
        logger.debug(f"[lifespan] stream stop error (non-fatal): {_se}")

    if broadcast_task is not None:
        broadcast_task.cancel()
    if stream_task is not None:
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass
    if broadcast_task is not None:
        try:
            await broadcast_task
        except asyncio.CancelledError:
            pass
    if scheduler is not None:
        scheduler.shutdown(wait=False)
    logger.info("Engine stopped — scheduler, stream, and broadcast loop shut down")


app = FastAPI(title="SignalBolt Engine", version="3.1.0", lifespan=lifespan)

# ── Rate limiting ─────────────────────────────────────────────────────────────
# slowapi gives us per-IP token-bucket limits without needing Redis. State is
# per-machine — with 2 app machines, a client gets 2x the limit in the worst
# case, which is acceptable for protecting Stripe/Supabase from abuse.
#
# Only applied to endpoints that hit expensive downstreams or can be used to
# spam users. Read-only public endpoints (/health, /prices, /indices, /signals)
# are left alone — they're cheap and cacheable.
#
# Disable per-IP limits in tests / dev by setting RATE_LIMIT_ENABLED=false.
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

_RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"


def _rate_key(request: Request) -> str:
    """
    Prefer the real client IP from Fly's forwarded headers; fall back to
    the socket address. Avoids treating every request as "the Fly LB" when
    behind a proxy.
    """
    fwd = request.headers.get("fly-client-ip") or request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(
    key_func=_rate_key,
    enabled=_RATE_LIMIT_ENABLED,
    # Default for any endpoint that opts in without specifying its own limit
    default_limits=["120/minute"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Fix #11: restrict CORS — allow_origins=["*"] is too broad for production.
# React Native/Expo mobile apps are not browsers so CORS doesn't apply to them,
# but web preview and development tooling do need explicit origins.
# Set ALLOWED_ORIGINS env var (comma-separated) to override defaults.
_default_origins = [
    "http://localhost:8081",    # Expo web dev server
    "http://localhost:19006",   # Expo web (older)
    "http://localhost:3000",    # any local web tooling
]
_env_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
_allowed_origins = _env_origins if _env_origins else _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",  # covers all Vercel preview deploys
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Threadpool sizing ─────────────────────────────────────────────────────────
# FastAPI runs sync `def` endpoints in anyio's threadpool (default 40 threads).
# Several endpoints (/prices, /indices, /signals, /history, /chart-data) are
# sync and make blocking Supabase/Alpaca calls. With the default limit, ~40
# concurrent slow requests will starve and 41+ will queue behind them.
# Bump to 100 — keeps memory modest (~100 MB) and absorbs burst load.
@app.on_event("startup")
async def _configure_threadpool() -> None:
    try:
        import anyio
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 100
        logger.info(f"[startup] anyio threadpool sized to {int(limiter.total_tokens)} threads")
    except Exception as e:
        logger.warning(f"[startup] threadpool resize failed (non-fatal): {e}")


# ── Hard per-request timeout ──────────────────────────────────────────────────
# Any individual request that runs longer than REQUEST_TIMEOUT_SECONDS is
# cancelled and the client gets HTTP 504. Without this, a hung Supabase/Alpaca
# call holds the worker thread forever — which is what caused our 60s+ /health
# blocks under load. /health and /ready are exempted so they always answer.
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))
_TIMEOUT_EXEMPT_PATHS   = {"/health", "/ready", "/ws/prices"}


@app.middleware("http")
async def request_timeout(request: Request, call_next):
    import asyncio
    if request.url.path in _TIMEOUT_EXEMPT_PATHS:
        return await call_next(request)
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        from fastapi.responses import JSONResponse
        logger.warning(
            f"[timeout] {request.method} {request.url.path} exceeded "
            f"{REQUEST_TIMEOUT_SECONDS}s — returning 504"
        )
        return JSONResponse(
            status_code=504,
            content={"error": "request_timeout", "limit_seconds": REQUEST_TIMEOUT_SECONDS},
        )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Skip /health entirely — Fly hits it every 15s and logging here adds noise
    # plus a tiny but real CPU cost on a path that must be fastest possible.
    if request.url.path == "/health":
        return await call_next(request)

    start    = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.3f}s"
    )
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# ── Market status (public — app uses it to render the holiday banner) ────────
_market_status_cache: dict = {}
_market_status_ts: float   = 0.0
_MARKET_STATUS_TTL: int    = 30  # 30s — short enough that the banner appears within 30s of session changes


@app.get("/market/status")
async def market_status():
    """
    Public market session snapshot. App calls this to decide whether to
    render a 'market closed' banner on signal-related screens.

    Cached 30s — calling /market/status from N tabs shouldn't hammer the
    calendar lookup. Returns shape:
      {
        is_open_today:  bool,
        is_open_now:    bool,
        mode:           "STANDARD" | "PRE_MARKET" | "AFTER_HOURS" | "BLOCKED" | ...,
        block_reason:   "NYSE holiday (2026-05-25) — market closed all day" | "",
        is_early_close: bool,
        close_et:       "16:00" | "13:00" | null   (when market is open today)
      }
    """
    global _market_status_cache, _market_status_ts
    now = time.monotonic()
    if now - _market_status_ts < _MARKET_STATUS_TTL and _market_status_cache:
        return _market_status_cache

    from engine.session_classifier import (
        is_market_open_today, is_market_open_now, today_close_mins_et,
        _is_early_close, classify,
    )

    sess = classify(has_premarket_catalyst=False)
    is_today = is_market_open_today()
    close_mins = today_close_mins_et() if is_today else None
    close_et   = f"{close_mins//60:02d}:{close_mins%60:02d}" if close_mins else None

    result = {
        "is_open_today":  is_today,
        "is_open_now":    is_market_open_now(),
        "mode":           sess["mode"],
        "block_reason":   sess["block_reason"],
        "is_early_close": _is_early_close(),
        "close_et":       close_et,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }
    _market_status_cache = result
    _market_status_ts    = now
    return result


@app.get("/earnings/calendar")
async def earnings_calendar(tickers: str = ""):
    """
    Weekly earnings calendar (Mon→Fri of the current week).

    Source: Finnhub free tier (requires FINNHUB_API_KEY env var).
    If the key isn't set, returns source="unavailable" and an empty list
    so the app can render a setup hint instead of crashing.

    Query params:
      tickers — optional comma-separated whitelist to filter results
                (e.g. "AAPL,NVDA,TSLA"). Omit for the full US calendar.

    Cached 1h via engine.cache (shared Redis when available).
    """
    from engine.earnings_service import get_weekly_earnings
    tlist = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
    # get_weekly_earnings is sync (httpx.Client). Run it off the event loop
    # so a slow Finnhub call doesn't block other requests.
    import anyio
    return await anyio.to_thread.run_sync(get_weekly_earnings, tlist)


@app.get("/health")
async def health():
    """
    Dependency-free liveness probe. MUST stay fast and offline-safe.

    Fly health checks point at this endpoint. It must never call Supabase,
    Alpaca, yfinance, Stripe, or Anthropic — any of those can hang for 10s+
    and previously caused Fly's load balancer to mark the VM unhealthy
    (PR04 "could not find a good candidate") even though the process was up.

    For deep dependency checks, use /ready.
    """
    return {
        "status":    "ok",
        "service":   "signalbolt-engine",
        "version":   "3.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/ready")
async def ready():
    """
    Deep readiness probe: verifies Supabase + Alpaca + key configuration.

    Uses strict timeouts so a slow downstream cannot block the response.
    Returns 'ready' when core deps respond, 'degraded' when at least one
    fails. Never raises — failures are reported in the body.
    """
    checks: dict[str, str] = {}
    overall = "ready"

    # ── Supabase ──────────────────────────────────────────────────────────────
    try:
        from engine.config import SUPABASE_URL, SUPABASE_SECRET_KEY
        sb = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
        sb.table("signals").select("id").limit(1).execute()
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"error: {e}"
        overall = "degraded"

    # ── Alpaca (strict 3s timeout) ────────────────────────────────────────────
    try:
        import httpx
        from engine.config import ALPACA_API_KEY, ALPACA_SECRET_KEY
        alpaca_base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        resp = httpx.get(
            f"{alpaca_base}/v2/account",
            headers={
                "APCA-API-KEY-ID":     ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            },
            timeout=3.0,
        )
        checks["alpaca"] = "healthy" if resp.status_code == 200 else f"http_{resp.status_code}"
        if resp.status_code != 200:
            overall = "degraded"
    except Exception as e:
        checks["alpaca"] = f"error: {e}"
        overall = "degraded"

    # ── Config-only checks (no network) ───────────────────────────────────────
    checks["anthropic"] = "configured" if os.environ.get("ANTHROPIC_API_KEY") else "missing"
    checks["stripe"]    = "configured" if os.environ.get("STRIPE_SECRET_KEY") else "missing"

    # ── Worker heartbeat ──────────────────────────────────────────────────────
    # Engine worker writes engine_heartbeats every 60s. If we haven't seen a
    # beat in WORKER_STALE_AFTER_SEC, the worker is silently dead — no signal
    # scans, no Alpaca stream, no push notifications. Surfaces here so Sentry /
    # uptime alerts can fire even though /health stays green.
    try:
        stale_after = int(os.environ.get("WORKER_STALE_AFTER_SEC", "300"))
        service_name = os.environ.get("WORKER_SERVICE_NAME", "engine_worker")
        sb_url = os.environ.get("SUPABASE_URL")
        sb_key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SECRET_KEY")
        if sb_url and sb_key:
            sb = create_client(sb_url, sb_key)
            rows = (
                sb.table("engine_heartbeats")
                .select("last_beat, machine_id")
                .eq("service", service_name)
                .limit(1)
                .execute()
                .data
            )
            if not rows:
                checks["worker"] = "unknown: no heartbeat yet"
            else:
                last_iso = rows[0]["last_beat"]
                # Supabase returns ISO with offset; parse robustly
                last = datetime.fromisoformat(last_iso.replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - last).total_seconds()
                if age <= stale_after:
                    checks["worker"] = f"healthy (last_beat {int(age)}s ago)"
                else:
                    checks["worker"] = f"stale ({int(age)}s ago, threshold {stale_after}s)"
                    overall = "degraded"
    except Exception as e:
        # Missing table is the common case before the migration runs — report
        # as "skipped" instead of "error" so it doesn't trigger alerts.
        msg = str(e).lower()
        if "engine_heartbeats" in msg and ("does not exist" in msg or "schema cache" in msg):
            checks["worker"] = "skipped: engine_heartbeats table missing (run migration)"
        else:
            checks["worker"] = f"error: {e}"

    return {
        "status":    overall,
        "service":   "signalbolt-engine",
        "version":   "3.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks":    checks,
    }


@app.get("/prices")
def get_prices(tickers: str):
    """
    Return live prices for comma-separated tickers.

    Priority chain (same feed the signal engine uses):
      1. Alpaca SIP snapshot  — real-time, covers pre/post market, all exchanges
      2. Polygon snapshot      — fallback if Alpaca fails (Polygon key optional)
      3. yfinance              — last resort, per-symbol, delayed

    Response shape per symbol:
      {
        price:                  float,   current price
        changePercent:          float,   vs prev close
        session:                str,     "pre"|"market"|"post"|"closed"
        extendedPrice:          float?,  pre/post market price
        extendedChangePercent:  float?,  pre/post market change %
      }

    Results are cached per ticker for 5 s so that multiple signal cards
    polling simultaneously don't hammer Alpaca on every request.

    Each ticker entry includes a `staleAfter` ISO timestamp. The UI should
    grey out the row once `now > staleAfter`. During market hours this is
    ~15 s out (one cache TTL). When the market is closed (weekends,
    holidays, or after 4 PM ET on a regular session) the snapshot is the
    last print — staleAfter is set to the next market open so the UI
    doesn't flash "stale" between every refresh.
    """
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    # Decide the staleAfter horizon once per request.
    # Open market   → "fresh for one cache TTL" (15 s)
    # Closed market → much longer; the print isn't going to change soon
    try:
        from engine.session_classifier import is_market_open_now
        _mkt_open = bool(is_market_open_now())
    except Exception:
        _mkt_open = True   # fail-open: never make the UI think data is stale on a transient error
    _now_dt = datetime.now(timezone.utc)
    _stale_after_iso = (
        _now_dt + timedelta(seconds=int(_PRICES_CACHE_TTL))
        if _mkt_open else
        _now_dt + timedelta(minutes=30)
    ).isoformat()

    # ── Serve cached entries; collect symbols that need a fresh fetch ──
    # Cache is now Redis-backed (cross-machine) when REDIS_URL is set, with
    # automatic fallback to per-process memory. The old _prices_cache dict
    # is kept as a second-level local mirror for sub-millisecond hits.
    from engine.cache import kv as _kv
    ttl_int = int(_PRICES_CACHE_TTL)
    now      = time.monotonic()
    result:  dict = {}
    to_fetch: list[str] = []
    for sym in symbols:
        # L1: process-local dict (fastest)
        local = _prices_cache.get(sym)
        if local and (now - local[0]) < _PRICES_CACHE_TTL:
            result[sym] = local[1]
            continue
        # L2: shared Redis (across machines)
        shared = _kv.get_json(f"prices:{sym}")
        if shared:
            result[sym] = shared
            _prices_cache[sym] = (now, shared)   # promote into L1
            continue
        to_fetch.append(sym)

    if to_fetch:
        # ── 1. Alpaca SIP (real-time, consistent with signal engine) ──
        fresh = _alpaca_stock_snapshots(to_fetch)

        # ── 2. Polygon fallback for any symbols Alpaca missed ─────────
        missing = [s for s in to_fetch if s not in fresh]
        if missing:
            poly = _polygon_stock_snapshots(missing)
            fresh.update(poly)

        # ── 3. yfinance last resort, one by one ───────────────────────
        still_missing = [s for s in to_fetch if s not in fresh]
        for sym in still_missing:
            data = _yf_price(sym)
            if data:
                fresh[sym] = data

        # ── Store in BOTH cache layers and merge into result ──────────
        ts = time.monotonic()
        for sym, data in fresh.items():
            _prices_cache[sym] = (ts, data)
            _kv.set_json(f"prices:{sym}", data, ttl_sec=ttl_int)
        result.update(fresh)

    # Stamp staleAfter on every entry (cached + freshly fetched alike) so
    # the UI has a uniform contract. Non-destructive: spread into a copy
    # so we don't mutate the cached dicts.
    return {sym: {**data, "staleAfter": _stale_after_iso} for sym, data in result.items()}


@app.websocket("/ws/prices")
async def ws_prices(websocket: WebSocket):
    """
    Real-time price stream via WebSocket.

    Protocol:
      1. Client connects → sends {"subscribe": ["SPY", "AAPL", ...]}
      2. Server replies immediately with current snapshot for those tickers
      3. Server pushes {"TICKER": {price, changePercent, session}} on every
         trade from Alpaca (throttled to max ~6/sec per ticker)
      4. Server sends {"ping": true} every 25 s to keep the connection alive

    The app replaces polling with this endpoint for real-time price display.
    """
    import asyncio as _asyncio
    from engine import price_store

    await websocket.accept()
    # maxsize=50 = 5 seconds of buffering at 10 Hz.
    # broadcast_snapshot() puts ONE batched message per cycle (not one per ticker),
    # so 50 slots is far more than enough. Old 200 was for the per-ticker design
    # where 20+ entries arrived every 100 ms and starvation caused burst/drop cycles.
    queue: _asyncio.Queue = _asyncio.Queue(maxsize=50)
    tickers: set[str] = set()

    try:
        # ── Step 1: receive subscription list ─────────────────────────────
        raw = await _asyncio.wait_for(websocket.receive_text(), timeout=10)
        msg = json.loads(raw)
        tickers = {t.strip().upper() for t in msg.get("subscribe", []) if t.strip()}

        price_store.add_client(queue, tickers)

        # ── Dynamic Alpaca subscription for custom tickers ─────────────────
        # Tickers not in ALL_TICKERS are not subscribed on the Alpaca stream at
        # startup. Subscribe them now so the stream delivers tick-by-tick updates
        # for custom watchlist symbols — no REST polling fallback needed.
        try:
            from engine.runner import ALL_TICKERS as _ALL_TICKERS
            from engine.stream import subscribe_extra_tickers
            extra = [t for t in tickers if t not in set(_ALL_TICKERS)]
            if extra:
                _asyncio.create_task(subscribe_extra_tickers(extra))
        except Exception as _dyn_e:
            logger.debug(f"[ws/prices] dynamic subscribe setup error: {_dyn_e}")

        # ── Step 2: send current snapshot immediately ──────────────────────
        snap = price_store.snapshot(list(tickers))

        # Fall back to REST price chain for tickers not yet in the live store
        # (price_store is empty outside market hours / before first trade arrives)
        missing = [t for t in tickers if t not in snap]
        if missing:
            try:
                rest_prices = await _asyncio.get_event_loop().run_in_executor(
                    None, get_prices, ",".join(missing)
                )
                snap.update(rest_prices)
                # Seed the price store AND prev_close so future WS trade ticks
                # compute changePercent correctly for custom watchlist tickers.
                for sym, data in rest_prices.items():
                    price_store.seed(
                        sym,
                        data["price"],
                        data["changePercent"],
                        data.get("session", "closed"),
                    )
                    # Derive prev_close: price / (1 + changePercent/100)
                    chg = data.get("changePercent", 0.0)
                    if chg != -100 and data["price"] > 0:
                        prev = data["price"] / (1 + chg / 100)
                        price_store.set_prev_close(sym, prev)
            except Exception as _fe:
                logger.debug(f"[ws/prices] REST fallback error: {_fe}")

        if snap:
            await websocket.send_text(json.dumps(snap))

        # ── Step 3: stream live updates + handle re-subscribes ───────────────
        # Two concurrent tasks run in parallel:
        #   • dequeue price updates from broadcast_snapshot and forward them
        #   • receive incoming WS messages from the app (re-subscribe requests)
        #
        # Why re-subscribe matters: the app sends a second {"subscribe":[...]}
        # when signals finish loading (visibleTickers changes after the initial
        # empty-set connect). Without this loop the server silently drops it.
        async def _recv_loop():
            """Handle incoming messages from the app (re-subscribe, etc.)."""
            while True:
                try:
                    raw_in = await websocket.receive_text()
                    msg_in = json.loads(raw_in)
                    new_subs = {
                        t.strip().upper()
                        for t in msg_in.get("subscribe", [])
                        if t.strip()
                    }
                    if new_subs:
                        # Merge with existing subscription set
                        tickers.update(new_subs)
                        price_store.add_client(queue, tickers)   # updates registry
                        # Subscribe new tickers on Alpaca if not already streaming
                        try:
                            from engine.runner import ALL_TICKERS as _ALL2
                            from engine.stream import subscribe_extra_tickers as _sub2
                            extra2 = [t for t in new_subs if t not in set(_ALL2)]
                            if extra2:
                                _asyncio.create_task(_sub2(extra2))
                        except Exception:
                            pass
                        # Send immediate snapshot for newly subscribed tickers
                        new_snap = price_store.snapshot(list(new_subs))
                        if new_snap:
                            await websocket.send_text(json.dumps(new_snap))
                except Exception:
                    break   # disconnect or parse error — let outer handler clean up

        async def _send_loop():
            """Forward price updates from broadcast_snapshot to the client."""
            while True:
                try:
                    update = await _asyncio.wait_for(queue.get(), timeout=25)
                    await websocket.send_text(update)
                except _asyncio.TimeoutError:
                    # Keepalive ping so the connection stays alive
                    await websocket.send_text(json.dumps({"ping": True}))

        # Run both loops concurrently — cancel both when either exits
        done, pending = await _asyncio.wait(
            [
                _asyncio.create_task(_recv_loop()),
                _asyncio.create_task(_send_loop()),
            ],
            return_when=_asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    except (WebSocketDisconnect, _asyncio.CancelledError):
        pass
    except Exception as e:
        logger.debug(f"[ws/prices] client error: {e}")
    finally:
        price_store.remove_client(queue)


@app.get("/indices")
def get_indices():
    """
    Returns SPY, QQQ, BTC, VIX live data plus derived Fear/Greed and VIX Sentiment.
    SPY/QQQ: Alpaca SIP primary → Polygon → yfinance.
    BTC: Polygon crypto primary → yfinance.
    VIX: yfinance only (Alpaca doesn't carry ^VIX).

    Result is cached for 5 seconds so the app can poll every 5 s without
    hammering Alpaca/yfinance on every request from every connected user.
    """
    global _indices_cache, _indices_cache_ts
    import time as _time
    from engine.cache import kv as _kv

    # L1: process-local dict (sub-ms hits)
    if _indices_cache and (_time.monotonic() - _indices_cache_ts) < _INDICES_CACHE_TTL:
        return _indices_cache

    # L2: shared Redis (cross-machine — prevents N machines × 1 Alpaca call)
    shared = _kv.get_json("indices:all")
    if shared:
        _indices_cache    = shared
        _indices_cache_ts = _time.monotonic()
        return shared

    result: dict = {}

    # ── SPY, QQQ via Alpaca SIP (real-time, same feed as signals) ──
    alp  = _alpaca_stock_snapshots(["SPY", "QQQ"])
    poly = _polygon_stock_snapshots(["SPY", "QQQ"]) if not alp else {}
    for sym in ("SPY", "QQQ"):
        result[sym] = alp.get(sym) or poly.get(sym) or _yf_price(sym) or {"price": 0, "changePercent": 0}

    # ── BTC via Alpaca crypto (24/7, free) → Polygon → yfinance ──
    btc = _alpaca_btc_snapshot() or _polygon_crypto_snapshot("X:BTCUSD") or _yf_price("BTC-USD")
    result["BTC"] = btc or {"price": 0, "changePercent": 0}

    # ── VIX via yfinance (Polygon doesn't carry ^VIX on standard plans) ──
    result["VIX"] = _yf_price("^VIX") or {"price": 20.0, "changePercent": 0}

    # ── Derived: Fear & Greed (inverse of VIX) ──
    vix = result["VIX"]["price"]
    if vix < 12:
        fg_score, fg_label = 85, "Extreme Greed"
    elif vix < 15:
        fg_score, fg_label = 68, "Greed"
    elif vix < 20:
        fg_score, fg_label = 50, "Neutral"
    elif vix < 25:
        fg_score, fg_label = 30, "Fear"
    elif vix < 30:
        fg_score, fg_label = 18, "Extreme Fear"
    else:
        fg_score, fg_label = 8,  "Panic"
    result["FEAR_GREED"] = {"score": fg_score, "label": fg_label}

    # ── Derived: VIX Sentiment ──
    if vix < 15:
        vix_sentiment = "BULLISH"
    elif vix < 20:
        vix_sentiment = "NEUTRAL"
    elif vix < 25:
        vix_sentiment = "CAUTIOUS"
    else:
        vix_sentiment = "BEARISH"
    result["VIX_SENTIMENT"] = {"label": vix_sentiment}

    # Store in BOTH cache layers
    _indices_cache    = result
    _indices_cache_ts = _time.monotonic()
    _kv.set_json("indices:all", result, ttl_sec=int(_INDICES_CACHE_TTL))

    return result


@app.get("/premarket")
def get_premarket(min_score: int = 0, limit: int = 50):
    """
    Return today's pre-market watchlist sorted by watch_score descending.

    Query params:
      min_score  — only return tickers with watch_score >= this (default 0)
      limit      — max rows (default 50)

    Falls back to live in-memory cache when Supabase is slow; returns []
    when neither the cache nor DB is populated yet (before 8 AM ET).
    """
    try:
        from engine import premarket_scanner as _pm
        cached = _pm._cache
        if cached is not None:
            rows = [
                vars(r) for r in cached.results
                if r.watch_score >= min_score
            ][:limit]
            return {"premarket": rows, "count": len(rows), "source": "cache"}

        # Fall back to Supabase DB
        sb        = _make_supabase()
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        query = (
            sb.table("premarket_watchlist")
            .select("*")
            .eq("scan_date", today_str)
            .gte("watch_score", min_score)
            .order("watch_score", desc=True)
            .limit(limit)
        )
        result = query.execute()
        return {"premarket": result.data, "count": len(result.data), "source": "db"}
    except Exception as e:
        logger.error(f"GET /premarket error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals")
async def get_signals(user_id: str = "", strategy_type: str = ""):
    """
    Return recent active signals.

    Hit by every app launch and every signals-tab refresh — by far the highest
    QPS endpoint. Converted to async Supabase so each request doesn't tie up an
    anyio threadpool thread for the ~50–200ms Supabase round-trip.
    """
    try:
        sb    = await _make_supabase_async()
        query = sb.table("signals").select("*").order("created_at", desc=True)
        if strategy_type:
            query = query.eq("strategy_type", strategy_type)
        result = await query.limit(50).execute()
        return {"signals": result.data, "count": len(result.data)}
    except Exception as e:
        logger.error(f"GET /signals error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RunRequest(BaseModel):
    tickers: Optional[List[str]] = None


def _check_engine_key(request: Request) -> None:
    """
    Fix #3: protect internal engine endpoints (/run, /inject-test-signal)
    from public access. Requires X-Engine-Key header or ?api_key= query param
    to match ENGINE_API_KEY env variable.
    If ENGINE_API_KEY is not set, endpoint is blocked entirely in production.
    """
    if not ENGINE_API_KEY:
        if ENVIRONMENT == "production":
            raise HTTPException(status_code=503, detail="Engine key not configured")
        return  # allow in dev/staging without a key set
    key = (
        request.headers.get("X-Engine-Key")
        or request.headers.get("Authorization", "").removeprefix("Bearer ")
        or request.query_params.get("api_key", "")
    )
    if key != ENGINE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid engine API key")


@app.post("/run")
@limiter.limit("3/minute")
def manual_run(req: RunRequest, background_tasks: BackgroundTasks, request: Request):
    _check_engine_key(request)
    from engine.runner import run_scan
    tickers = [t.upper() for t in req.tickers] if req.tickers else None
    background_tasks.add_task(run_scan, tickers=tickers)
    return {
        "status": "triggered",
        "message": "Signal scan started in background",
        "tickers": tickers or "default watchlist",
    }


@app.get("/history/{ticker}")
def get_history(ticker: str, from_ts: str = "", to_ts: str = "", interval: str = "1h"):
    """
    Return OHLC price history for a ticker between two ISO timestamps.
    Used by the signal replay chart in the app.
    """
    try:
        import yfinance as yf
        from datetime import timedelta

        # Default window: last 7 days if no timestamps given
        if from_ts and to_ts:
            start = from_ts[:10]
            end   = to_ts[:10]
        else:
            from datetime import date
            end   = date.today().isoformat()
            start = (date.today() - timedelta(days=7)).isoformat()

        df = yf.download(
            ticker.upper(),
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return {"candles": [], "ticker": ticker}

        df = df.reset_index()
        date_col = "Datetime" if "Datetime" in df.columns else "Date"
        candles = [
            {
                "t": str(row[date_col]),
                "o": round(float(row["Open"]),  2),
                "h": round(float(row["High"]),  2),
                "l": round(float(row["Low"]),   2),
                "c": round(float(row["Close"]), 2),
            }
            for _, row in df.iterrows()
        ]
        return {"candles": candles, "ticker": ticker}
    except Exception as e:
        logger.error(f"GET /history/{ticker} error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chart-data/{ticker}")
def get_chart_data(ticker: str, timeframe: str = "15Min", bars: int = 60):
    """
    Return OHLCV candlestick bars for the signal chart modal.
    Uses Alpaca for accurate intraday data; falls back to yfinance.

    timeframe: "5Min" | "15Min" | "1Hour" | "1Day"
    bars: number of candles to return (default 60)
    """

    ticker = ticker.upper()

    # ── Alpaca path ────────────────────────────────────────────────────────────
    try:
        from engine.alpaca_client import get_bars as alpaca_get_bars

        tf_to_days = {"5Min": 2, "15Min": 3, "1Hour": 7, "1Day": 60}
        days = tf_to_days.get(timeframe, 3)

        df = alpaca_get_bars(ticker, timeframe=timeframe, days=days)

        if df is not None and not df.empty:
            # Keep last N bars only
            df = df.tail(bars).reset_index()

            ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]

            candles = []
            for _, row in df.iterrows():
                ts = row[ts_col]
                # Convert to ISO string
                if hasattr(ts, "isoformat"):
                    ts_str = ts.isoformat()
                else:
                    ts_str = str(ts)
                candles.append({
                    "t": ts_str,
                    "o": round(float(row["open"]),  2),
                    "h": round(float(row["high"]),  2),
                    "l": round(float(row["low"]),   2),
                    "c": round(float(row["close"]), 2),
                    "v": int(row["volume"]) if "volume" in row else 0,
                })

            return {"candles": candles, "ticker": ticker, "timeframe": timeframe, "source": "alpaca"}
    except Exception as e:
        logger.warning(f"[chart-data] Alpaca path failed for {ticker}: {e} — trying yfinance")

    # ── yfinance fallback ──────────────────────────────────────────────────────
    try:
        import yfinance as yf
        from datetime import timedelta, date

        tf_yf_map   = {"5Min": "5m",  "15Min": "15m", "1Hour": "1h",  "1Day": "1d"}
        tf_days_map  = {"5Min": 2,    "15Min": 5,     "1Hour": 20,    "1Day": 90}
        yf_interval  = tf_yf_map.get(timeframe, "15m")
        yf_days      = tf_days_map.get(timeframe, 5)

        end   = date.today().isoformat()
        start = (date.today() - timedelta(days=yf_days)).isoformat()

        df = yf.download(ticker, start=start, end=end, interval=yf_interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return {"candles": [], "ticker": ticker, "timeframe": timeframe, "source": "none"}

        df = df.tail(bars).reset_index()
        date_col = "Datetime" if "Datetime" in df.columns else "Date"

        candles = []
        for _, row in df.iterrows():
            ts = row[date_col]
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            candles.append({
                "t":  ts_str,
                "o":  round(float(row["Open"]),   2),
                "h":  round(float(row["High"]),   2),
                "l":  round(float(row["Low"]),    2),
                "c":  round(float(row["Close"]),  2),
                "v":  int(row["Volume"]) if "Volume" in row else 0,
            })

        return {"candles": candles, "ticker": ticker, "timeframe": timeframe, "source": "yfinance"}

    except Exception as e:
        logger.error(f"GET /chart-data/{ticker} fallback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class PushTokenRequest(BaseModel):
    user_id: str
    push_token: str


@app.post("/register-push")
def register_push_token(req: PushTokenRequest):
    """Save a device's Expo push token to the profiles table."""
    try:
        sb = _make_supabase()
        sb.table("profiles").update({"push_token": req.push_token}).eq("id", req.user_id).execute()
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"POST /register-push error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Stripe endpoints
# ---------------------------------------------------------------------------

@app.get("/checkout-success")
def checkout_success(plan: str = "pro", session_id: str = ""):
    from fastapi.responses import HTMLResponse

    # Verify payment with Stripe and update DB directly — no webhook needed
    if session_id:
        try:
            import traceback as _tb
            session  = stripe.checkout.Session.retrieve(session_id)
            if session.payment_status == "paid":
                meta      = session.metadata or {}
                user_id   = meta["user_id"]  if "user_id" in meta else ""
                paid_plan = meta["plan"]      if "plan"    in meta else plan
                if user_id:
                    sb = _make_supabase()
                    update_data: dict = {
                        "subscription_status": paid_plan,
                        "tier":                paid_plan,
                        "free_ends_at":        None,
                    }
                    if session.customer:
                        update_data["stripe_customer_id"] = session.customer
                    sb.table("profiles").update(update_data).eq("id", user_id).execute()
                    logger.info(f"[stripe] Profile updated via success page user={user_id} plan={paid_plan}")
        except Exception as e:
            logger.error(f"[stripe] Success page DB update failed: {e}\n{_tb.format_exc()}")

    label = "Pro+" if plan == "pro_plus" else "Pro"
    return HTMLResponse(f"""
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Payment Successful — SignalBolt</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;display:flex;align-items:center;
justify-content:center;min-height:100vh;margin:0;background:#f9fafb}}
.card{{text-align:center;padding:40px 32px;max-width:400px;width:100%}}
.icon{{font-size:64px;margin-bottom:16px}}
h1{{color:#00C805;font-size:28px;margin:0 0 8px;font-weight:700}}
p{{color:#6b7280;font-size:16px;margin:0 0 28px;line-height:1.5}}
.badge{{background:#dcfce7;color:#15803d;padding:6px 16px;border-radius:20px;
font-size:14px;font-weight:600;display:inline-block;margin-bottom:28px}}
.btn{{display:inline-block;background:#00C805;color:#fff;text-decoration:none;
padding:14px 32px;border-radius:12px;font-size:16px;font-weight:600;
border:none;cursor:pointer;width:100%;box-sizing:border-box}}
.hint{{color:#9ca3af;font-size:13px;margin-top:16px}}
</style>
</head>
<body><div class="card">
<div class="icon">⚡</div>
<h1>You're on {label}!</h1>
<div class="badge">✓ Payment confirmed</div>
<p>Your subscription is active and ready to go.</p>
<button class="btn" onclick="window.location.href='signalbolt://checkout-success?plan={plan}'">
  Return to SignalBolt
</button>
<p class="hint">Or simply switch back to the app — it will update automatically.</p>
</div></body></html>""")


@app.get("/billing-return")
def billing_return():
    from fastapi.responses import HTMLResponse
    return HTMLResponse("""
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SignalBolt</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;display:flex;align-items:center;
justify-content:center;min-height:100vh;margin:0;background:#f9fafb}
.card{text-align:center;padding:40px 32px;max-width:400px;width:100%}
.icon{font-size:56px;margin-bottom:16px}
h1{color:#111827;font-size:24px;margin:0 0 8px;font-weight:700}
p{color:#6b7280;font-size:15px;margin:0 0 28px;line-height:1.5}
.btn{display:block;background:#111827;color:#fff;text-decoration:none;
padding:14px 32px;border-radius:12px;font-size:15px;font-weight:600;
border:none;cursor:pointer;width:100%;box-sizing:border-box}
.hint{color:#9ca3af;font-size:12px;margin-top:14px}
</style>
</head>
<body><div class="card">
<div class="icon">↩</div>
<h1>All done!</h1>
<p>Your billing changes have been saved.</p>
<p style="color:#111827;font-weight:600;font-size:16px">Switch back to the SignalBolt app — it will update automatically.</p>
</div></body></html>""")


@app.get("/checkout-cancel")
def checkout_cancel():
    from fastapi.responses import HTMLResponse
    return HTMLResponse("""
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cancelled — SignalBolt</title>
<style>body{{font-family:-apple-system,sans-serif;display:flex;align-items:center;
justify-content:center;min-height:100vh;margin:0;background:#f9fafb}}
.card{{text-align:center;padding:40px;max-width:400px}}
.icon{{font-size:64px;margin-bottom:16px}}
h1{{color:#374151;font-size:28px;margin:0 0 8px}}
p{{color:#6b7280;font-size:16px;margin:0}}</style></head>
<body><div class="card">
<div class="icon">↩</div>
<h1>Payment cancelled</h1>
<p>No charge was made. Return to the app and try again whenever you're ready.</p>
</div></body></html>""")


@app.post("/create-payment-intent")
@limiter.limit("5/minute")
async def create_payment_intent(request: Request):
    """Create a SetupIntent + ephemeral key for the native Stripe payment sheet."""
    body    = await request.json()
    user_id = body.get("user_id")
    email   = body.get("email")
    plan    = body.get("plan")

    if not user_id or not plan:
        raise HTTPException(status_code=400, detail="user_id and plan are required")

    price_id = STRIPE_PRO_PRICE_ID if plan == "pro" else STRIPE_PRO_PLUS_PRICE_ID
    if not price_id:
        raise HTTPException(status_code=500, detail=f"Price ID not configured for plan: {plan}")

    try:
        # Find or create Stripe customer
        customers      = stripe.Customer.list(email=email, limit=1)
        customer_list  = customers.to_dict().get("data", [])
        if customer_list:
            customer_id = customer_list[0].get("id")
        else:
            customer    = stripe.Customer.create(email=email, metadata={"user_id": user_id})
            customer_id = customer.id
            # Persist customer ID immediately
            _make_supabase().table("profiles").update({"stripe_customer_id": customer_id}).eq("id", user_id).execute()

        ephemeral_key = stripe.EphemeralKey.create(customer=customer_id, stripe_version="2023-10-16")
        setup_intent  = stripe.SetupIntent.create(
            customer=customer_id,
            payment_method_types=["card"],
            usage="off_session",
            metadata={"user_id": user_id, "plan": plan, "price_id": price_id},
        )
        logger.info(f"[stripe] SetupIntent created for user={user_id} plan={plan}")
        return {
            "client_secret":   setup_intent.client_secret,
            "customer_id":     customer_id,
            "ephemeral_key":   ephemeral_key.secret,
            "publishable_key": os.getenv("STRIPE_PUBLISHABLE_KEY", ""),
        }
    except stripe.StripeError as e:
        logger.error(f"POST /create-payment-intent Stripe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"POST /create-payment-intent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create-subscription-after-payment")
@limiter.limit("5/minute")
async def create_subscription_after_payment(request: Request):
    """After setup intent confirmed in app, attach the saved card and create subscription."""
    body             = await request.json()
    user_id          = body.get("user_id")
    customer_id      = body.get("customer_id")
    payment_method_id = body.get("payment_method_id", "")
    plan             = body.get("plan")

    if not user_id or not customer_id or not plan:
        raise HTTPException(status_code=400, detail="user_id, customer_id, and plan are required")

    price_id = STRIPE_PRO_PRICE_ID if plan == "pro" else STRIPE_PRO_PLUS_PRICE_ID
    if not price_id:
        raise HTTPException(status_code=500, detail=f"Price ID not configured for plan: {plan}")

    try:
        # If client didn't send payment_method_id, find the most recently attached one
        if not payment_method_id:
            pms = stripe.PaymentMethod.list(customer=customer_id, type="card")
            pm_list = pms.to_dict().get("data", [])
            if not pm_list:
                raise HTTPException(status_code=400, detail="No payment method found for customer")
            payment_method_id = pm_list[0].get("id")

        # Set as default on customer
        stripe.Customer.modify(customer_id, invoice_settings={"default_payment_method": payment_method_id})

        # Cancel any existing active subscriptions so user only ever has one
        existing_subs = stripe.Subscription.list(customer=customer_id, limit=10)
        for sub in existing_subs.to_dict().get("data", []):
            if sub.get("status") in ("active", "trialing"):
                stripe.Subscription.cancel(sub.get("id"))
                logger.info(f"[stripe] Cancelled existing subscription={sub.get('id')} before creating new one")

        # Create the subscription
        subscription = stripe.Subscription.create(
            customer=customer_id,
            items=[{"price": price_id}],
            default_payment_method=payment_method_id,
            metadata={"user_id": user_id, "plan": plan},
        )
        sub_dict = subscription.to_dict()
        status   = sub_dict.get("status", "")

        if status in ("active", "trialing"):
            from datetime import timezone as _tz
            sb = _make_supabase()
            sb.table("profiles").update({
                "subscription_status":    plan,
                "tier":                   plan,
                "stripe_customer_id":     customer_id,
                "free_ends_at":           None,
                "subscription_synced_at": datetime.now(_tz.utc).isoformat(),
            }).eq("id", user_id).execute()
            logger.info(f"[stripe] Subscription created plan={plan} user={user_id}")
            return {"status": "success", "subscription_status": plan}
        else:
            logger.warning(f"[stripe] Subscription created but status={status} user={user_id}")
            return {"status": "failed", "stripe_status": status}

    except HTTPException:
        raise
    except stripe.StripeError as e:
        logger.error(f"POST /create-subscription-after-payment Stripe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"POST /create-subscription-after-payment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create-checkout")
@limiter.limit("5/minute")
async def create_checkout(request: Request):
    """Create a Stripe Checkout session and return the URL."""
    body = await request.json()
    user_id = body.get("user_id")
    email   = body.get("email")
    plan    = body.get("plan")

    if not user_id or not plan:
        raise HTTPException(status_code=400, detail="user_id and plan are required")

    price_id = STRIPE_PRO_PRICE_ID if plan == "pro" else STRIPE_PRO_PLUS_PRICE_ID
    if not price_id:
        raise HTTPException(status_code=500, detail=f"Price ID not configured for plan: {plan}")

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            customer_email=email,
            line_items=[{"price": price_id, "quantity": 1}],
            metadata={"user_id": user_id, "plan": plan},
            success_url=f"signalbolt://payment-success?plan={plan}&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url="signalbolt://payment-cancelled",
            allow_promotion_codes=True,
        )
        logger.info(f"[stripe] Checkout session created for user={user_id} plan={plan}")
        return {"url": session.url}
    except Exception as e:
        logger.error(f"POST /create-checkout error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync-subscription")
@limiter.limit("10/minute")
async def sync_subscription(request: Request):
    """Check Stripe for the current subscription status and update the profile."""
    import time as _time
    body    = await request.json()
    user_id = body.get("user_id")
    email   = body.get("email", "")
    force   = body.get("force", False)
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    try:
        sb = _make_supabase()

        res = sb.table("profiles").select("email, subscription_status, subscription_synced_at").eq("id", user_id).single().execute()
        row = res.data or {}
        if not email:
            email = row.get("email", "")

        current_status = row.get("subscription_status", "free")

        # Skip Stripe call if synced within 6 hours and not forced
        if not force:
            synced_at = row.get("subscription_synced_at")
            if synced_at:
                from datetime import timezone as _tz
                age_seconds = (datetime.now(_tz.utc) - datetime.fromisoformat(synced_at.replace("Z", "+00:00"))).total_seconds()
                if age_seconds < 6 * 3600:
                    logger.info(f"[stripe] Sync skipped (cached {int(age_seconds/60)}min ago) user={user_id}")
                    return {"subscription_status": current_status, "cached": True}

        # Find Stripe customer by email
        customers = stripe.Customer.list(email=email, limit=1)
        customers_dict = customers.to_dict()
        customer_list  = customers_dict.get("data", [])

        if not customer_list:
            if current_status != "free":
                sb.table("profiles").update({"subscription_status": "free"}).eq("id", user_id).execute()
                logger.info(f"[stripe] Sync: no customer found, set free user={user_id}")
            return {"subscription_status": "free", "reason": "no_stripe_customer"}

        customer_id = customer_list[0].get("id")
        # Save customer ID for billing portal lookups
        sb.table("profiles").update({"stripe_customer_id": customer_id}).eq("id", user_id).execute()

        # Fetch subscriptions using to_dict() for safe access
        subscriptions = stripe.Subscription.list(customer=customer_id, limit=10, expand=["data.items.data.price"])
        subs_dict = subscriptions.to_dict()
        subs_list = subs_dict.get("data", [])

        logger.info(f"[stripe] Sync found {len(subs_list)} subscription(s) for user={user_id}")

        # Separate genuinely active subs from cancel_at_period_end ones
        genuine_active = None
        cancelling_sub = None
        for sub in subs_list:
            status = sub.get("status")
            if status in ("active", "trialing"):
                if sub.get("cancel_at_period_end"):
                    if cancelling_sub is None:
                        cancelling_sub = sub
                else:
                    genuine_active = sub
                    break  # found a real active sub — stop looking

        logger.info(f"[stripe] Sync: genuine_active={genuine_active is not None} cancelling={cancelling_sub is not None} user={user_id}")

        if genuine_active:
            # Genuinely active subscription
            items_data = genuine_active.get("items", {}).get("data", [{}])
            price_id   = items_data[0].get("price", {}).get("id", "") if items_data else ""
            new_status = "pro_plus" if price_id == STRIPE_PRO_PLUS_PRICE_ID else "pro"
            logger.info(f"[stripe] Sync: active sub price={price_id} → {new_status} user={user_id}")
        elif cancelling_sub:
            # Cancelled but still within billing period — revoke access immediately
            new_status = "expired"
            logger.info(f"[stripe] Sync: cancel_at_period_end → expired user={user_id}")
        else:
            new_status = "expired"
            logger.info(f"[stripe] Sync: no active subscription → expired user={user_id}")

        # Always stamp synced_at so cache TTL resets after a real Stripe call
        from datetime import timezone as _tz
        update_payload: dict = {"subscription_synced_at": datetime.now(_tz.utc).isoformat()}
        if new_status != current_status:
            update_payload["subscription_status"] = new_status
            update_payload["tier"] = new_status
            if new_status in ("pro", "pro_plus"):
                update_payload["free_ends_at"] = None
            logger.info(f"[stripe] Synced {current_status}→{new_status} user={user_id}")
        else:
            logger.info(f"[stripe] Sync confirmed unchanged status={new_status} user={user_id}")
        sb.table("profiles").update(update_payload).eq("id", user_id).execute()

        return {"subscription_status": new_status, "cached": False}

    except stripe.StripeError as e:
        logger.error(f"POST /sync-subscription Stripe error: {e}")
        raise HTTPException(status_code=500, detail=f"Stripe error: {e}")
    except Exception as e:
        logger.error(f"POST /sync-subscription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create-portal")
@limiter.limit("5/minute")
async def create_portal(request: Request):
    """Stripe Customer Portal — accepts return_url from client."""
    body       = await request.json()
    user_id    = body.get("user_id")
    email      = body.get("email", "")
    return_url = body.get("return_url", f"{ENGINE_PUBLIC_URL}/billing-return")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    try:
        sb  = _make_supabase()
        res = sb.table("profiles").select("stripe_customer_id, email").eq("id", user_id).single().execute()
        row = res.data or {}
        customer_id = row.get("stripe_customer_id")
        if not customer_id:
            email = email or row.get("email", "")
            if email:
                customers = stripe.Customer.list(email=email, limit=1)
                cd = customers.to_dict().get("data", [])
                if cd:
                    customer_id = cd[0].get("id")
                    sb.table("profiles").update({"stripe_customer_id": customer_id}).eq("id", user_id).execute()
        if not customer_id:
            raise HTTPException(status_code=404, detail="No billing account found. Complete a purchase first.")
        portal = stripe.billing_portal.Session.create(customer=customer_id, return_url=return_url)
        logger.info(f"[stripe] Portal session created for user={user_id}")
        return {"url": portal.url}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /create-portal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# Fix #10: /create-billing-portal was a duplicate of /create-portal.
# Removed — use /create-portal which also accepts a return_url parameter.

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events to update subscription status."""
    payload    = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid signature")

    sb = _make_supabase()

    if event["type"] == "checkout.session.completed":
        session     = event["data"]["object"]
        user_id     = session["metadata"]["user_id"]
        plan        = session["metadata"]["plan"]
        customer_id = session["customer"]
        sb.table("profiles").update({
            "subscription_status": plan,
            "tier":                plan,
            "stripe_customer_id":  customer_id,
            "free_ends_at":        None,
        }).eq("id", user_id).execute()
        logger.info(f"[stripe] Subscription activated user={user_id} plan={plan}")

    elif event["type"] == "customer.subscription.deleted":
        customer_id = event["data"]["object"]["customer"]
        sb.table("profiles").update({
            "subscription_status": "expired",
            "tier":                "expired",
        }).eq("stripe_customer_id", customer_id).execute()
        logger.info(f"[stripe] Subscription cancelled customer={customer_id}")

    elif event["type"] == "customer.subscription.updated":
        obj         = event["data"]["object"]
        customer_id = obj["customer"]
        # Cancelled at period end — mark expired immediately so app reflects it
        if obj["cancel_at_period_end"]:
            sb.table("profiles").update({
                "subscription_status": "expired",
                "tier":                "expired",
            }).eq("stripe_customer_id", customer_id).execute()
            logger.info(f"[stripe] Subscription set to cancel at period end customer={customer_id}")
        else:
            # Reactivated or plan changed
            meta = obj["metadata"] if "metadata" in obj else {}
            plan = meta["plan"] if "plan" in meta else ""
            if plan in ("pro", "pro_plus"):
                sb.table("profiles").update({
                    "subscription_status": plan,
                    "tier":                plan,
                }).eq("stripe_customer_id", customer_id).execute()
                logger.info(f"[stripe] Subscription updated customer={customer_id} plan={plan}")

    elif event["type"] == "invoice.payment_failed":
        customer_id = event["data"]["object"]["customer"]
        logger.warning(f"[stripe] Payment failed customer={customer_id}")

    return {"status": "ok"}


@app.post("/cancel-subscription")
@limiter.limit("5/minute")
async def cancel_subscription(request: Request):
    """Cancel the user's active Stripe subscription at period end."""
    body       = await request.json()
    user_id    = body.get("user_id")
    immediately = body.get("immediately", False)
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    try:
        sb  = _make_supabase()
        res = sb.table("profiles").select("stripe_customer_id, email").eq("id", user_id).single().execute()
        row = res.data or {}
        customer_id = row.get("stripe_customer_id")
        if not customer_id:
            email = row.get("email", "")
            if email:
                customers = stripe.Customer.list(email=email, limit=1)
                cd = customers.to_dict().get("data", [])
                if cd:
                    customer_id = cd[0].get("id")
        if not customer_id:
            raise HTTPException(status_code=404, detail="No billing account found")

        subs = stripe.Subscription.list(customer=customer_id, limit=10)
        subs_list = subs.to_dict().get("data", [])
        active_sub = next((s for s in subs_list if s.get("status") in ("active", "trialing") and not s.get("cancel_at_period_end")), None)
        if not active_sub:
            raise HTTPException(status_code=404, detail="No active subscription found")

        sub_id = active_sub.get("id")
        from datetime import timezone as _tz

        if immediately:
            stripe.Subscription.cancel(sub_id)
            # Cut off access now
            sb.table("profiles").update({
                "subscription_status":    "expired",
                "subscription_synced_at": datetime.now(_tz.utc).isoformat(),
            }).eq("id", user_id).execute()
            logger.info(f"[stripe] Subscription cancelled immediately user={user_id}")
        else:
            stripe.Subscription.modify(sub_id, cancel_at_period_end=True)
            # Keep current plan active — Stripe webhook will set expired when period ends
            sb.table("profiles").update({
                "subscription_synced_at": datetime.now(_tz.utc).isoformat(),
            }).eq("id", user_id).execute()
            logger.info(f"[stripe] Subscription set to cancel at period end user={user_id}")

        return {"status": "cancelled", "immediately": immediately}
    except HTTPException:
        raise
    except stripe.StripeError as e:
        logger.error(f"POST /cancel-subscription Stripe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"POST /cancel-subscription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-account")
@limiter.limit("3/minute")
async def delete_account(request: Request):
    """
    Permanently delete the authenticated user's account.
    Requires a valid Supabase Bearer token in Authorization header.
    Deletes: auth.users row (via admin API) + signals + profiles.
    """
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="No auth token provided")
    try:
        sb = _make_supabase()
        # Verify token and get user ID
        user_resp = sb.auth.get_user(token)
        if not user_resp or not user_resp.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = user_resp.user.id

        # Delete profile data (signals cascade if FK set up, else delete explicitly)
        sb.table("profiles").delete().eq("id", user_id).execute()

        # Delete auth user using admin API (requires service role key)
        sb.auth.admin.delete_user(user_id)

        logger.info(f"Account deleted for user {user_id}")
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DELETE /delete-account error: {e}")
        raise HTTPException(status_code=500, detail="Account deletion failed")


@app.get("/invoices")
async def get_invoices(user_id: str):
    """Return the last 10 Stripe invoices for the user."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    try:
        sb  = _make_supabase()
        res = sb.table("profiles").select("stripe_customer_id, email").eq("id", user_id).single().execute()
        row = res.data or {}
        customer_id = row.get("stripe_customer_id")
        if not customer_id:
            email = row.get("email", "")
            if email:
                customers = stripe.Customer.list(email=email, limit=1)
                cd = customers.to_dict().get("data", [])
                if cd:
                    customer_id = cd[0].get("id")
        if not customer_id:
            return {"invoices": []}

        invoices = stripe.Invoice.list(customer=customer_id, limit=10)
        result = []
        for inv in invoices.to_dict().get("data", []):
            result.append({
                "id":          inv.get("id"),
                "amount":      inv.get("amount_paid", 0),
                "currency":    inv.get("currency", "usd"),
                "status":      inv.get("status"),
                "date":        inv.get("created"),
                "pdf":         inv.get("invoice_pdf"),
                "description": inv.get("description") or "Subscription",
            })
        return {"invoices": result}
    except Exception as e:
        logger.error(f"GET /invoices error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-payment-method")
@limiter.limit("5/minute")
async def update_payment_method(request: Request):
    """After new SetupIntent confirmed, set latest payment method as default on subscription."""
    body        = await request.json()
    user_id     = body.get("user_id")
    customer_id = body.get("customer_id")
    if not user_id or not customer_id:
        raise HTTPException(status_code=400, detail="user_id and customer_id are required")
    try:
        pms    = stripe.PaymentMethod.list(customer=customer_id, type="card")
        pm_list = pms.to_dict().get("data", [])
        if not pm_list:
            raise HTTPException(status_code=400, detail="No payment method found")
        pm_id = pm_list[0].get("id")

        stripe.Customer.modify(customer_id, invoice_settings={"default_payment_method": pm_id})

        subs = stripe.Subscription.list(customer=customer_id, limit=5)
        for sub in subs.to_dict().get("data", []):
            if sub.get("status") in ("active", "trialing"):
                stripe.Subscription.modify(sub.get("id"), default_payment_method=pm_id)
                break

        logger.info(f"[stripe] Payment method updated user={user_id}")
        return {"status": "updated"}
    except HTTPException:
        raise
    except stripe.StripeError as e:
        logger.error(f"POST /update-payment-method Stripe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"POST /update-payment-method error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Test / dev helpers
# ---------------------------------------------------------------------------

@app.post("/inject-test-signal")
async def inject_test_signal(request: Request):
    _check_engine_key(request)
    """
    Inserts a realistic sample signal with full quant metadata directly into
    Supabase. Useful for testing the app UI without waiting for market hours.
    Accepts optional overrides in the JSON body.

    Example body (all fields optional):
      { "ticker": "NVDA", "direction": "LONG", "strategy": "day_trade",
        "scenario": "t1_hit" }

    scenario options:
      "fresh"     — new active signal (default)
      "t1_hit"    — stop already moved to breakeven (tests the B/E badge)
      "win"       — closed signal with target hit (tests Activity Feed)
      "loss"      — closed signal stopped out (tests Activity Feed)
    """
    from datetime import timezone as _tz
    import random, uuid

    body      = {}
    try:
        body  = await request.json()
    except Exception:
        pass

    ticker    = (body.get("ticker")    or "NVDA").upper()
    direction = (body.get("direction") or "LONG").upper()
    strategy  = body.get("strategy")  or "day_trade"
    scenario  = body.get("scenario")  or "fresh"

    # Realistic price levels relative to a fake entry
    entry   = round(random.uniform(180, 220), 2)
    sl_pct  = 0.018   # 1.8% stop
    t1_pct  = 0.025   # 2.5% target 1
    t2_pct  = 0.05    # 5.0% target 2

    if direction == "LONG":
        stop_loss  = round(entry * (1 - sl_pct), 2)
        target_one = round(entry * (1 + t1_pct), 2)
        target_two = round(entry * (1 + t2_pct), 2)
    else:
        stop_loss  = round(entry * (1 + sl_pct), 2)
        target_one = round(entry * (1 - t1_pct), 2)
        target_two = round(entry * (1 - t2_pct), 2)

    rr = round(abs(target_one - entry) / abs(entry - stop_loss), 2)

    now_utc = datetime.now(_tz.utc)

    row: dict = {
        "ticker":            ticker,
        "direction":         direction,
        "entry_price":       entry,
        "stop_loss":         stop_loss,
        "target_one":        target_one,
        "target_two":        target_two,
        "confidence_score":  random.randint(78, 94),
        "ai_explanation":    (
            f"Price swept buy-side liquidity and formed a strong order block at ${entry:.2f}. "
            f"SMC structure is {direction.lower()}ish with bullish CHoCH on the 15m. "
            f"VIX is low, ADX confirms trend, and no manipulation patterns detected. "
            f"Gamma wall at ${target_two:.2f} acts as a natural ceiling — TP2 set just below."
        ),
        "timeframe":         "15m",
        "strategy_type":     strategy,
        "status":            "active",
        "result":            None,
        "result_pnl":        None,
        "result_pct":        None,
        "closed_at":         None,
        "closed_reason":     None,
        "hit_target":        False,
        "created_at":        now_utc.isoformat(),
        # ── Quant metadata ──────────────────────────────────────────────
        "confidence_tier":    "A",
        "position_multiplier": 0.75,
        "regime_type":        "TRENDING_BULL",
        "session_mode":       "STANDARD",
        "gamma_net_gex":      1_250_000,
        "gamma_is_negative":  False,
        "manipulation_clean": True,
        "manipulation_flags": [],
        "sl_adjustments":     ["gamma_support_buffer", "atr_volatility_adj"],
        "risk_reward":        rr,
        "score_breakdown": {
            "l1_smc":        22.0,
            "l2_technical":  18.5,
            "l3_sentiment":  14.0,
            "l4_risk":       12.0,
            "l5_mtf":         8.0,
            "l6_regime":      7.0,
            "l7_session":     4.0,
            "l8_gamma":       6.5,
            "l9_manipulation": 6.0,
        },
    }

    # ── Apply scenario overrides ─────────────────────────────────────────
    if scenario == "t1_hit":
        # Stop already moved to breakeven — triggers the B/E badge in the app
        row["stop_loss"] = entry

    elif scenario == "win":
        row["status"]       = "closed"
        row["result"]       = "win"
        row["result_pct"]   = round(t1_pct * 100, 2)
        row["result_pnl"]   = round(entry * t1_pct, 2)
        row["hit_target"]   = True
        row["closed_at"]    = now_utc.isoformat()
        row["closed_reason"] = "target_hit"

    elif scenario == "loss":
        row["status"]       = "closed"
        row["result"]       = "loss"
        row["result_pct"]   = round(-sl_pct * 100, 2)
        row["result_pnl"]   = round(-entry * sl_pct, 2)
        row["hit_target"]   = False
        row["closed_at"]    = now_utc.isoformat()
        row["closed_reason"] = "stop_hit"

    try:
        sb  = _make_supabase()
        res = sb.table("signals").insert(row).execute()
        sig = res.data[0] if res.data else {}
        sig_id = sig.get("id")

        # ── Write scenario events to signal_events ───────────────────────────
        # The DB trigger auto-writes the 'fired' event on insert.
        # We add additional events here for non-fresh scenarios.
        if sig_id:
            extra_events = []
            if scenario == "t1_hit":
                extra_events.append({
                    "signal_id":  sig_id,
                    "event_type": "t1_hit",
                    "price":      entry * (1 + t1_pct),
                    "note":       f"T1 hit @ ${entry*(1+t1_pct):.2f} (+{t1_pct*100:.1f}%) — stop moved to breakeven ${entry:.2f}",
                })
            elif scenario == "win":
                extra_events.append({
                    "signal_id":  sig_id,
                    "event_type": "t1_hit",
                    "price":      entry * (1 + t1_pct),
                    "note":       f"T1 hit @ ${entry*(1+t1_pct):.2f} (+{t1_pct*100:.1f}%) — stop moved to breakeven",
                })
                extra_events.append({
                    "signal_id":  sig_id,
                    "event_type": "closed_win",
                    "price":      entry * (1 + t1_pct),
                    "note":       f"Target 1 hit — closed +{t1_pct*100:.1f}%",
                })
            elif scenario == "loss":
                extra_events.append({
                    "signal_id":  sig_id,
                    "event_type": "closed_loss",
                    "price":      entry * (1 - sl_pct),
                    "note":       f"Stop Loss hit — stopped out {-sl_pct*100:.1f}%",
                })
            if extra_events:
                try:
                    sb.table("signal_events").insert(extra_events).execute()
                except Exception as ev_err:
                    logger.warning(f"[inject] Event logging skipped: {ev_err}")

        # Send push notification for active signals
        if scenario in ("fresh", "t1_hit"):
            try:
                from engine.push import send_signal_alert, _send_raw
                if scenario == "t1_hit":
                    _send_raw(
                        title=f"🎯 T1 Hit — {ticker} +{t1_pct*100:.1f}%",
                        body=f"Stop moved to breakeven. Riding to T2. {direction} still open.",
                        data={"type": "t1_breakeven", "ticker": ticker},
                    )
                else:
                    send_signal_alert(ticker, direction, row["confidence_score"])
            except Exception as push_err:
                logger.warning(f"[inject] Push notification skipped: {push_err}")

        logger.info(f"[inject] Test signal inserted — {ticker} {direction} {strategy} scenario={scenario} id={sig_id}")
        return {
            "status":   "inserted",
            "scenario": scenario,
            "signal":   sig,
        }

    except Exception as e:
        logger.error(f"POST /inject-test-signal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/close-signal")
async def admin_close_signal(request: Request):
    """
    Force-close any signal by ID. Useful for bad-data signals (wrong entry
    price, erroneous Alpaca tick) that should never have fired.

    Requires ENGINE_API_KEY header (X-Engine-Key).

    Body:
      { "id": "<signal-uuid>",
        "reason": "bad_data" | "manual" | "bad_entry" | "regime_change" | ...
        "result": "void" | "loss" | "win"   (optional, default "void") }

    Example curl:
      curl -X POST https://signalbolt-engine.fly.dev/admin/close-signal \\
           -H "X-Engine-Key: YOUR_KEY" \\
           -H "Content-Type: application/json" \\
           -d '{"id":"abc-123","reason":"bad_data"}'
    """
    _check_engine_key(request)

    body = {}
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON body required")

    signal_id = (body.get("id") or "").strip()
    if not signal_id:
        raise HTTPException(status_code=400, detail="'id' is required")

    reason = (body.get("reason") or "manual").strip()
    result = (body.get("result") or "void").strip()

    try:
        sb = _make_supabase()

        # Fetch signal first so we can log what we're closing
        existing = sb.table("signals").select(
            "id, ticker, direction, strategy_type, entry_price, status"
        ).eq("id", signal_id).single().execute()

        if not existing.data:
            raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")

        sig = existing.data
        if sig["status"] == "closed":
            return {
                "status":  "already_closed",
                "signal":  sig,
                "message": "Signal was already closed — no change made",
            }

        # Close the signal
        update = {
            "status":        "closed",
            "result":        result,
            "closed_reason": reason,
            "closed_at":     datetime.now(timezone.utc).isoformat(),
        }
        sb.table("signals").update(update).eq("id", signal_id).execute()

        # Write a timeline event so there's a clear audit trail
        sb.table("signal_events").insert({
            "signal_id":  signal_id,
            "event_type": "admin_closed",
            "price":      sig["entry_price"],
            "note":       f"Admin force-closed — reason: {reason} | result: {result}",
        }).execute()

        logger.info(
            f"[admin] Force-closed signal {signal_id} "
            f"{sig['ticker']} {sig['direction']} — reason={reason} result={result}"
        )

        return {
            "status":  "closed",
            "reason":  reason,
            "result":  result,
            "signal":  {**sig, **update},
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /admin/close-signal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/active-signals")
async def admin_active_signals(request: Request):
    """
    List all currently active signals with their entry prices and current
    prices so bad-data signals are easy to spot at a glance.

    Requires ENGINE_API_KEY header (X-Engine-Key).
    """
    _check_engine_key(request)

    try:
        sb = await _make_supabase_async()
        rows = (await (
            sb.table("signals")
            .select("id, ticker, direction, strategy_type, entry_price, target_one, target_two, stop_loss, confidence_score, created_at")
            .eq("status", "active")
            .order("created_at", desc=True)
            .execute()
        )).data or []

        # Enrich with current price from price_store (no extra API call)
        from engine import price_store
        enriched = []
        for r in rows:
            current = price_store.snapshot([r["ticker"]]).get(r["ticker"], {})
            current_price = current.get("price")
            entry         = float(r["entry_price"])
            deviation_pct = (
                round(abs(current_price - entry) / entry * 100, 1)
                if current_price else None
            )
            enriched.append({
                **r,
                "current_price":   current_price,
                "deviation_pct":   deviation_pct,
                "likely_bad_data": deviation_pct is not None and deviation_pct > 20,
            })

        return {
            "count":   len(enriched),
            "signals": enriched,
        }

    except Exception as e:
        logger.error(f"GET /admin/active-signals error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ── Premium Feature Endpoints ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

from engine.config import (
    ENABLE_HEATMAP, ENABLE_QUANT_DASHBOARD,
    ENABLE_NEWS_REACTION, ENABLE_SOCIAL_SIGNALS,
)


def _require_jwt(request: Request):
    """
    Validate Supabase Bearer token and return (user_id, sb).
    Raises 401 on missing or invalid token.
    """
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header required")
    sb = _make_supabase()
    try:
        user_resp = sb.auth.get_user(token)
        if not user_resp or not user_resp.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_resp.user.id, sb
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Token verification failed")


# ── Market Heatmap ────────────────────────────────────────────────────────────

@app.get("/market/heatmap")
async def market_heatmap(
    request: Request,
    sort_by: str = "momentum",
    filter_by: Optional[str] = None,
    sector: Optional[str] = None,
    min_rel_volume: float = 0.0,
):
    """
    Real-time market heatmap — momentum, volume, trend direction per ticker.
    Available to all authenticated users (free + paid).
    sort_by: momentum | gainers | losers | volume
    filter_by: bullish | bearish | high_volume | None
    """
    if not ENABLE_HEATMAP:
        raise HTTPException(status_code=503, detail="Heatmap feature is disabled")

    # Verify JWT (any authenticated user can see heatmap)
    _require_jwt(request)

    try:
        from engine.heatmap_service import compute_heatmap
        sb = _make_supabase()

        # Pull active signals to highlight tickers with live signals
        sig_rows = (
            sb.table("signals")
            .select("ticker, id")
            .eq("status", "active")
            .execute()
            .data or []
        )
        active_signals = {r["ticker"]: r["id"] for r in sig_rows}

        data = compute_heatmap(
            symbols=None,
            active_signals=active_signals,
            sort_by=sort_by,
            filter_by=filter_by,
            sector=sector,
            min_rel_volume=min_rel_volume,
        )
        return {"tickers": data, "count": len(data)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GET /market/heatmap error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Quant Dashboard ───────────────────────────────────────────────────────────

@app.get("/quant/dashboard")
async def quant_dashboard(request: Request):
    """
    Retail-friendly quant score dashboard.
    Returns market regime + 6 stock-screening buckets with setup ratings.
    Available to Pro and Pro+ users.
    """
    if not ENABLE_QUANT_DASHBOARD:
        raise HTTPException(status_code=503, detail="Quant dashboard is disabled")

    _require_jwt(request)

    try:
        from engine.quant_score_service import get_quant_dashboard
        return get_quant_dashboard()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GET /quant/dashboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── News Reaction Feed ────────────────────────────────────────────────────────

@app.get("/news/reaction")
async def news_reaction(request: Request, limit: int = 20):
    """
    Real-time news reaction feed — headline sentiment, price reaction, urgency.
    Links news items to any active signals on the same ticker.
    Available to all authenticated users.
    """
    if not ENABLE_NEWS_REACTION:
        raise HTTPException(status_code=503, detail="News reaction feature is disabled")

    _require_jwt(request)

    try:
        from engine.news_reaction_service import get_news_feed
        from engine.heatmap_service import DEFAULT_TICKERS
        sb = _make_supabase()

        # Build ticker → signalId map for linkage
        sig_rows = (
            sb.table("signals")
            .select("ticker, id")
            .eq("status", "active")
            .execute()
            .data or []
        )
        active_signals = {r["ticker"]: r["id"] for r in sig_rows}

        items = get_news_feed(
            tickers=DEFAULT_TICKERS,
            active_signals=active_signals,
            limit=limit,
        )
        return {"items": items, "count": len(items)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GET /news/reaction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Community / Social Signal Feed ────────────────────────────────────────────

@app.get("/signals/community")
async def community_feed(request: Request, limit: int = 20):
    """
    Community feed: active signals sorted by community interest score.
    Returns vote counts, follow counts, comments, communityScore.
    """
    if not ENABLE_SOCIAL_SIGNALS:
        raise HTTPException(status_code=503, detail="Social signals feature is disabled")

    _require_jwt(request)

    try:
        from engine.community_service import get_community_feed
        sb = _make_supabase()
        feed = get_community_feed(sb, limit=limit)
        return {"feed": feed, "count": len(feed)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GET /signals/community error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/{signal_id}/social-summary")
async def signal_social_summary(signal_id: str, request: Request):
    """Social summary (votes, follows, comments, score) for one signal."""
    if not ENABLE_SOCIAL_SIGNALS:
        raise HTTPException(status_code=503, detail="Social signals feature is disabled")

    _require_jwt(request)

    try:
        from engine.community_service import get_social_summary
        sb = _make_supabase()
        return get_social_summary(signal_id, sb)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GET /signals/{signal_id}/social-summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signals/{signal_id}/vote")
async def vote_signal(signal_id: str, request: Request):
    """
    Cast or update a vote on a signal.
    Body: { "vote_type": "bullish" | "bearish" | "watching" }
    """
    if not ENABLE_SOCIAL_SIGNALS:
        raise HTTPException(status_code=503, detail="Social signals feature is disabled")

    user_id, sb = _require_jwt(request)

    body = {}
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON body required")

    vote_type = (body.get("vote_type") or "").strip()
    if vote_type not in ("bullish", "bearish", "watching"):
        raise HTTPException(status_code=400, detail="vote_type must be bullish | bearish | watching")

    try:
        from engine.community_service import add_vote
        return add_vote(signal_id, user_id, vote_type, sb)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /signals/{signal_id}/vote error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signals/{signal_id}/comment")
async def comment_signal(signal_id: str, request: Request):
    """
    Add a comment to a signal.
    Body: { "content": "..." }
    Max 500 chars, max 5 comments per user per signal.
    """
    if not ENABLE_SOCIAL_SIGNALS:
        raise HTTPException(status_code=503, detail="Social signals feature is disabled")

    user_id, sb = _require_jwt(request)

    body = {}
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON body required")

    content = (body.get("content") or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")

    try:
        from engine.community_service import add_comment
        return add_comment(signal_id, user_id, content, sb)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /signals/{signal_id}/comment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/{signal_id}/comments")
async def get_signal_comments(signal_id: str, request: Request, limit: int = 20):
    """Return public (non-flagged) comments for a signal."""
    if not ENABLE_SOCIAL_SIGNALS:
        raise HTTPException(status_code=503, detail="Social signals feature is disabled")

    _require_jwt(request)

    try:
        from engine.community_service import get_comments
        sb = _make_supabase()
        return {"comments": get_comments(signal_id, sb, limit=limit)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GET /signals/{signal_id}/comments error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signals/{signal_id}/follow")
async def follow_signal(signal_id: str, request: Request):
    """Toggle follow/unfollow on a signal. Returns { following: bool, followCount: int }."""
    if not ENABLE_SOCIAL_SIGNALS:
        raise HTTPException(status_code=503, detail="Social signals feature is disabled")

    user_id, sb = _require_jwt(request)

    try:
        from engine.community_service import toggle_follow
        return toggle_follow(signal_id, user_id, sb)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /signals/{signal_id}/follow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signals/report-comment")
async def report_comment_endpoint(request: Request):
    """Flag a comment for moderation. Body: { 'comment_id': '...' }"""
    if not ENABLE_SOCIAL_SIGNALS:
        raise HTTPException(status_code=503, detail="Social signals feature is disabled")

    _require_jwt(request)

    body = {}
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON body required")

    comment_id = (body.get("comment_id") or "").strip()
    if not comment_id:
        raise HTTPException(status_code=400, detail="comment_id is required")

    try:
        from engine.community_service import report_comment
        sb = _make_supabase()
        report_comment(comment_id, sb)
        return {"status": "flagged"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /signals/report-comment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from engine.config import PORT
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
