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
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from typing import List, Optional

import sentry_sdk
import stripe
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

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
_INDICES_CACHE_TTL: float = 3.0   # seconds

# Per-ticker price cache so the app can poll every 5 s without hammering Alpaca.
# Keys are individual ticker symbols; each entry is (timestamp, price_dict).
_prices_cache: dict[str, tuple[float, dict]] = {}
_PRICES_CACHE_TTL: float = 3.0   # seconds


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


def _make_supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], _supabase_key())


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

@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    from engine.runner import start_scheduler
    from engine.stream import run_stream
    from engine import price_store

    # ── Initialise real-time price store with this event loop ────────────────
    price_store.init(asyncio.get_event_loop())

    # ── Seed price store from Alpaca REST snapshot so first WS connect gets
    #    immediate data even before the first trade arrives ───────────────────
    try:
        from engine.runner import ALL_TICKERS
        seed_tickers = list(dict.fromkeys(ALL_TICKERS))[:40]
        snaps = _alpaca_stock_snapshots(seed_tickers)
        for ticker, data in snaps.items():
            price_store.seed(ticker, data["price"], data["changePercent"], data.get("session", "market"))
            # Derive prev_close so trade-based Δ% is accurate:
            #   prev_close = price / (1 + changePercent/100)
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

    logger.info(
        "SignalBolt engine started — "
        "scalping=WebSocket real-time | day_trade/swing=APScheduler | "
        "prices=WebSocket push to app"
    )
    yield

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    stream_task.cancel()
    try:
        await stream_task
    except asyncio.CancelledError:
        pass
    scheduler.shutdown(wait=False)
    logger.info("Engine stopped — scheduler and stream shut down")


app = FastAPI(title="SignalBolt Engine", version="3.1.0", lifespan=lifespan)

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


@app.middleware("http")
async def log_requests(request: Request, call_next):
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

@app.get("/health")
async def health():
    from engine.config import SUPABASE_URL, SUPABASE_SECRET_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY

    # Database check
    try:
        sb = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
        sb.table("signals").select("id").limit(1).execute()
        db_status = "healthy"
    except Exception as e:
        db_status = f"error: {e}"
        if SENTRY_DSN:
            sentry_sdk.capture_exception(e)

    # Alpaca check — use direct HTTP with a short timeout to avoid blocking
    try:
        import httpx
        alpaca_base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        resp = httpx.get(
            f"{alpaca_base}/v2/account",
            headers={
                "APCA-API-KEY-ID":     ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            },
            timeout=5.0,
        )
        alpaca_status = "healthy" if resp.status_code == 200 else f"http_{resp.status_code}"
    except Exception as e:
        alpaca_status = f"error: {e}"

    return {
        "status":    "ok",
        "service":   "signalbolt-engine",
        "version":   "3.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "database":  db_status,
            "alpaca":    alpaca_status,
            "anthropic": "configured" if ANTHROPIC_API_KEY else "missing",
            "stripe":    "configured" if STRIPE_SECRET_KEY else "missing",
        },
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
    """
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    now     = time.monotonic()

    # ── Serve cached entries; collect symbols that need a fresh fetch ──
    result:  dict = {}
    to_fetch: list[str] = []
    for sym in symbols:
        cached = _prices_cache.get(sym)
        if cached and (now - cached[0]) < _PRICES_CACHE_TTL:
            result[sym] = cached[1]
        else:
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

        # ── Store in cache and merge into result ──────────────────────
        ts = time.monotonic()
        for sym, data in fresh.items():
            _prices_cache[sym] = (ts, data)
        result.update(fresh)

    return result


@app.websocket("/ws/prices")
async def ws_prices(websocket: WebSocket):
    """
    Real-time price stream via WebSocket.

    Protocol:
      1. Client connects → sends {"subscribe": ["SPY", "AAPL", ...]}
      2. Server replies immediately with current snapshot for those tickers
      3. Server pushes {"TICKER": {price, changePercent, session}} on every
         trade from Alpaca (throttled to max 2/sec per ticker)
      4. Server sends {"ping": true} every 25 s to keep the connection alive

    The app replaces polling with this endpoint for real-time price display.
    """
    import asyncio as _asyncio
    from engine import price_store

    await websocket.accept()
    queue: _asyncio.Queue = _asyncio.Queue(maxsize=200)
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
                asyncio.create_task(subscribe_extra_tickers(extra))
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
                # Also seed the price store so future WS connects get it instantly
                for sym, data in rest_prices.items():
                    price_store.seed(
                        sym,
                        data["price"],
                        data["changePercent"],
                        data.get("session", "closed"),
                    )
            except Exception as _fe:
                logger.debug(f"[ws/prices] REST fallback error: {_fe}")

        if snap:
            await websocket.send_text(json.dumps(snap))

        # ── Step 3: stream live updates ───────────────────────────────────
        while True:
            try:
                update = await _asyncio.wait_for(queue.get(), timeout=25)
                await websocket.send_text(update)
            except _asyncio.TimeoutError:
                # Keepalive ping so the connection stays open
                await websocket.send_text(json.dumps({"ping": True}))

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

    # Serve from cache if fresh
    if _indices_cache and (_time.monotonic() - _indices_cache_ts) < _INDICES_CACHE_TTL:
        return _indices_cache

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

    # Store in cache
    _indices_cache    = result
    _indices_cache_ts = _time.monotonic()

    return result


@app.get("/signals")
def get_signals(user_id: str = "", strategy_type: str = ""):
    """
    Return recent active signals.
    Fix #12: accepts optional user_id (for logging) and strategy_type filter.
    Signals are broadcast to all subscribers — no personal data here.
    """
    try:
        sb    = _make_supabase()
        query = sb.table("signals").select("*").order("created_at", desc=True)
        if strategy_type:
            query = query.eq("strategy_type", strategy_type)
        result = query.limit(50).execute()
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
            "stripe_customer_id":  customer_id,
            "free_ends_at":        None,
        }).eq("id", user_id).execute()
        logger.info(f"[stripe] Subscription activated user={user_id} plan={plan}")

    elif event["type"] == "customer.subscription.deleted":
        customer_id = event["data"]["object"]["customer"]
        sb.table("profiles").update({
            "subscription_status": "expired",
        }).eq("stripe_customer_id", customer_id).execute()
        logger.info(f"[stripe] Subscription cancelled customer={customer_id}")

    elif event["type"] == "customer.subscription.updated":
        obj         = event["data"]["object"]
        customer_id = obj["customer"]
        # Cancelled at period end — mark expired immediately so app reflects it
        if obj["cancel_at_period_end"]:
            sb.table("profiles").update({
                "subscription_status": "expired",
            }).eq("stripe_customer_id", customer_id).execute()
            logger.info(f"[stripe] Subscription set to cancel at period end customer={customer_id}")
        else:
            # Reactivated or plan changed
            meta = obj["metadata"] if "metadata" in obj else {}
            plan = meta["plan"] if "plan" in meta else ""
            if plan in ("pro", "pro_plus"):
                sb.table("profiles").update({
                    "subscription_status": plan,
                }).eq("stripe_customer_id", customer_id).execute()
                logger.info(f"[stripe] Subscription updated customer={customer_id} plan={plan}")

    elif event["type"] == "invoice.payment_failed":
        customer_id = event["data"]["object"]["customer"]
        logger.warning(f"[stripe] Payment failed customer={customer_id}")

    return {"status": "ok"}


@app.post("/cancel-subscription")
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


if __name__ == "__main__":
    import uvicorn
    from engine.config import PORT
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
