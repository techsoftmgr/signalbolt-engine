"""
Real-time price store — bridges the Alpaca trade stream to app WebSocket clients.

Flow:
  Alpaca trade → stream.py on_trade() → price_store.update()
              → throttled broadcast → each connected WS client queue
              → FastAPI /ws/prices endpoint → phone screen

Thread safety:
  update() is called from the Alpaca SDK thread (sync).
  Broadcasts are scheduled onto the FastAPI asyncio event loop via
  loop.call_soon_threadsafe so no asyncio primitives are touched from
  the wrong thread.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger("signalbolt.price_store")

ET = ZoneInfo("America/New_York")

# ── Price state ───────────────────────────────────────────────────────────────

# Latest price per ticker: {ticker: {price, changePercent, session}}
_prices: dict[str, dict] = {}

# Previous-day close per ticker — used to compute live changePercent from trades.
# Seeded from the Alpaca REST snapshot at startup.
_prev_close: dict[str, float] = {}

# ── WebSocket client registry ─────────────────────────────────────────────────

# Each connected client gets an asyncio.Queue.  Messages are JSON strings.
_clients: list[asyncio.Queue] = []
_client_tickers: dict[int, set[str]] = {}   # id(queue) → subscribed tickers

# ── Broadcast throttle ────────────────────────────────────────────────────────

_last_sent: dict[str, float] = {}
_THROTTLE_S: float = 0.5   # at most 2 price pushes per second per ticker

# ── Event loop reference ──────────────────────────────────────────────────────

_loop: asyncio.AbstractEventLoop | None = None


# ── Initialisation ────────────────────────────────────────────────────────────

def init(loop: asyncio.AbstractEventLoop) -> None:
    """Call once at FastAPI startup with the running event loop."""
    global _loop
    _loop = loop
    logger.info("[price_store] Initialised on event loop")


def set_prev_close(ticker: str, close: float) -> None:
    """Store previous-day close so changePercent can be computed from raw trades."""
    _prev_close[ticker] = close


def seed(ticker: str, price: float, change_pct: float, session: str) -> None:
    """
    Seed the store from a REST snapshot at startup.
    Ensures WebSocket clients that connect before the first trade
    still receive an immediate price response.
    """
    _prices[ticker] = {
        "price":         round(price, 2),
        "changePercent": round(change_pct, 3),
        "session":       session,
    }


# ── Called from Alpaca trade stream (sync thread) ────────────────────────────

def _market_session_now() -> str:
    """Simple time-based session tag — does not call Alpaca REST."""
    now = datetime.now(ET)
    h, m = now.hour, now.minute
    minutes = h * 60 + m
    if   minutes < 4 * 60:           return "closed"
    elif minutes < 9 * 60 + 30:      return "pre"
    elif minutes < 16 * 60:          return "market"
    elif minutes < 20 * 60:          return "post"
    else:                             return "closed"


def update(ticker: str, trade_price: float) -> None:
    """
    Update price from a live Alpaca trade.
    Called from the Alpaca SDK thread — must be thread-safe.
    """
    prev    = _prev_close.get(ticker)
    chg_pct = ((trade_price - prev) / prev * 100) if prev else 0.0
    session = _market_session_now()

    _prices[ticker] = {
        "price":         round(trade_price, 2),
        "changePercent": round(chg_pct, 3),
        "session":       session,
    }

    # Throttle: don't broadcast more than once per _THROTTLE_S per ticker
    now = time.monotonic()
    if now - _last_sent.get(ticker, 0) < _THROTTLE_S:
        return
    _last_sent[ticker] = now

    # Schedule async broadcast on the FastAPI event loop
    if _loop and not _loop.is_closed() and _clients:
        _loop.call_soon_threadsafe(
            lambda t=ticker, d=dict(_prices[ticker]):
                asyncio.ensure_future(_broadcast(t, d))
        )


# ── Async broadcast (runs on FastAPI event loop) ──────────────────────────────

async def _broadcast(ticker: str, data: dict) -> None:
    if not _clients:
        return
    msg = json.dumps({ticker: data})
    for q in list(_clients):
        subs = _client_tickers.get(id(q), set())
        if not subs or ticker in subs:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass   # slow client — skip this tick, they'll catch up


# ── WebSocket client management ───────────────────────────────────────────────

def add_client(q: asyncio.Queue, tickers: set[str]) -> None:
    _clients.append(q)
    _client_tickers[id(q)] = tickers
    logger.debug(f"[price_store] WS client added — {len(_clients)} connected, tickers={tickers}")


def remove_client(q: asyncio.Queue) -> None:
    try:
        _clients.remove(q)
    except ValueError:
        pass
    _client_tickers.pop(id(q), None)
    logger.debug(f"[price_store] WS client removed — {len(_clients)} remaining")


def snapshot(tickers: list[str]) -> dict:
    """Current prices for given tickers — for immediate send on WS connect."""
    return {t: _prices[t] for t in tickers if t in _prices}
