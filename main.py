import os
import logging
import requests
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "")


def _supabase_key() -> str:
    return os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]


def _make_supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], _supabase_key())


# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------

def _polygon_stock_snapshots(symbols: list[str]) -> dict:
    """Bulk stock snapshot from Polygon — price + day change for multiple tickers."""
    if not POLYGON_KEY or not symbols:
        return {}
    try:
        r = requests.get(
            "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers",
            params={"tickers": ",".join(symbols), "apiKey": POLYGON_KEY},
            timeout=8,
        )
        result = {}
        for t in r.json().get("tickers", []):
            sym   = t["ticker"]
            price = float(t.get("day", {}).get("c") or t.get("prevDay", {}).get("c") or 0)
            prev  = float(t.get("prevDay", {}).get("c") or price)
            chg   = float(t.get("todaysChangePerc") or (((price - prev) / prev * 100) if prev else 0))
            if price > 0:
                result[sym] = {"price": round(price, 2), "changePercent": round(chg, 2)}
        logger.debug(f"[polygon] got snapshots for {list(result.keys())}")
        return result
    except Exception as e:
        logger.debug(f"[polygon] stock snapshots error: {e}")
        return {}


def _polygon_crypto_snapshot(symbol: str) -> Optional[dict]:
    """Single crypto snapshot from Polygon (e.g. X:BTCUSD)."""
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
            return {"price": round(price, 2), "changePercent": round(chg, 2)}
    except Exception as e:
        logger.debug(f"[yfinance] {ticker}: {e}")
    return None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    from engine.runner import start_scheduler
    scheduler = start_scheduler()
    logger.info("SignalBolt engine started — scanning every 15 minutes")
    yield
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")


app = FastAPI(title="SignalBolt Engine", version="3.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "signalbolt-engine",
        "version": "3.1.0",
        "polygon": bool(POLYGON_KEY),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/prices")
def get_prices(tickers: str):
    """
    Return live prices for comma-separated tickers.
    Polygon is primary; yfinance is fallback for any missing symbol.
    """
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    result  = _polygon_stock_snapshots(symbols)

    # yfinance fallback for anything Polygon didn't return
    for sym in symbols:
        if sym not in result:
            data = _yf_price(sym)
            if data:
                result[sym] = data
    return result


@app.get("/indices")
def get_indices():
    """
    Returns SPY, QQQ, BTC, VIX live data plus derived Fear/Greed and VIX Sentiment.
    SPY/QQQ: Polygon primary. BTC: Polygon crypto primary. VIX: yfinance only.
    """
    result: dict = {}

    # ── SPY, QQQ via Polygon ──
    poly = _polygon_stock_snapshots(["SPY", "QQQ"])
    for sym in ("SPY", "QQQ"):
        result[sym] = poly.get(sym) or _yf_price(sym) or {"price": 0, "changePercent": 0}

    # ── BTC via Polygon crypto, fallback yfinance ──
    btc = _polygon_crypto_snapshot("X:BTCUSD")
    result["BTC"] = btc or _yf_price("BTC-USD") or {"price": 0, "changePercent": 0}

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

    return result


@app.get("/signals")
def get_signals():
    try:
        sb = _make_supabase()
        result = (
            sb.table("signals")
            .select("*")
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        return {"signals": result.data, "count": len(result.data)}
    except Exception as e:
        logger.error(f"GET /signals error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RunRequest(BaseModel):
    tickers: Optional[List[str]] = None


@app.post("/run")
def manual_run(req: RunRequest, background_tasks: BackgroundTasks):
    from engine.runner import run_scan
    tickers = [t.upper() for t in req.tickers] if req.tickers else None
    background_tasks.add_task(run_scan, tickers=tickers)
    return {
        "status": "triggered",
        "message": "Signal scan started in background",
        "tickers": tickers or "default watchlist",
    }
