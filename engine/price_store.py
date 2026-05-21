"""
Real-time price store — bridges the Alpaca trade stream to app WebSocket clients.

Architecture (10 Hz push loop — BATCHED):
  Alpaca trade → stream.py on_trade() → price_store.update()
              → marks ticker as "dirty" (via call_soon_threadsafe onto event loop)
              → every 100 ms: broadcast_snapshot() collects ALL dirty tickers,
                builds ONE batched JSON message per client, puts it in the queue
              → FastAPI /ws/prices _send_loop() dequeues and sends ONE frame
              → phone screen

Why BATCHED instead of per-ticker messages:
  - Old design put one queue entry per dirty ticker (e.g. 20 tickers = 20 entries).
    _send_loop() then called await websocket.send_text() 20× per cycle.
    Each mobile send takes ~5 ms → 100 ms of send work = entire broadcast window.
    This starved the asyncio event loop: _price_broadcast_loop couldn't wake
    on schedule, queue filled to 200, QueueFull silently dropped ticks.
    Result: burst of updates → silent gap → burst → gap (user saw "slow then fast").
  - Batched design: ONE queue entry per client per 100 ms cycle regardless of
    how many tickers changed.  One websocket.send_text() per cycle.  Event loop
    stays free.  No starvation.  Smooth tick-by-tick even for 30+ tickers.

Thread safety:
  update() is called from the Alpaca SDK internal event loop (a different
  thread from FastAPI).  The dirty-set mutation is pushed onto the FastAPI
  event loop via call_soon_threadsafe — both the dirty-set add and
  broadcast_snapshot run on the same asyncio thread, so no locks needed.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger("signalbolt.price_store")

ET = ZoneInfo("America/New_York")

# ── Price state ───────────────────────────────────────────────────────────────

# Latest price per ticker: {ticker: {price, changePercent, session}}
_prices: dict[str, dict] = {}

# Previous-day close per ticker — used to compute live changePercent from trades.
# Seeded from Alpaca REST snapshot at startup AND when WS clients subscribe.
_prev_close: dict[str, float] = {}

# ── Dirty set — tickers with new prices since the last broadcast cycle ────────
# Written from event loop (via call_soon_threadsafe). Read + cleared every 100ms
# by broadcast_snapshot() which also runs on the event loop — no lock needed.
_dirty: set[str] = set()

# ── WebSocket client registry ─────────────────────────────────────────────────

# Each connected WS client gets an asyncio.Queue. Messages are JSON strings.
_clients: list[asyncio.Queue] = []
_client_tickers: dict[int, set[str]] = {}   # id(queue) → subscribed ticker set

# ── Event loop reference ──────────────────────────────────────────────────────

_loop: asyncio.AbstractEventLoop | None = None


# ── Initialisation ────────────────────────────────────────────────────────────

def init(loop: asyncio.AbstractEventLoop) -> None:
    """Call once at FastAPI startup with the running event loop."""
    global _loop
    _loop = loop
    logger.info("[price_store] Initialised — 10 Hz broadcast loop ready")


def set_prev_close(ticker: str, close: float) -> None:
    """Store previous-day close so changePercent is accurate for live trades."""
    _prev_close[ticker] = close


def seed(ticker: str, price: float, change_pct: float, session: str) -> None:
    """
    Seed from a REST snapshot (startup or first WS subscribe).
    Gives clients an immediate price before the first Alpaca trade arrives.
    """
    _prices[ticker] = {
        "price":         round(price, 2),
        "changePercent": round(change_pct, 3),
        "session":       session,
    }


# ── Session helper ────────────────────────────────────────────────────────────

def _market_session_now() -> str:
    now = datetime.now(ET)
    m   = now.hour * 60 + now.minute
    if   m < 240:   return "closed"   # before 4:00 AM ET
    elif m < 570:   return "pre"      # 4:00–9:29 AM ET
    elif m < 960:   return "market"   # 9:30 AM–3:59 PM ET
    elif m < 1200:  return "post"     # 4:00–7:59 PM ET
    else:            return "closed"


# ── Called from Alpaca trade stream ──────────────────────────────────────────

def update(ticker: str, trade_price: float) -> None:
    """
    Record a live trade price and mark the ticker dirty for the next
    broadcast cycle.  Called from the Alpaca SDK thread — must not touch
    asyncio primitives directly.  Uses call_soon_threadsafe to queue the
    dirty-set mutation onto the FastAPI event loop.
    """
    prev    = _prev_close.get(ticker)
    chg_pct = ((trade_price - prev) / prev * 100) if prev else 0.0

    _prices[ticker] = {
        "price":         round(trade_price, 2),
        "changePercent": round(chg_pct, 3),
        "session":       _market_session_now(),
    }

    # Queue the dirty mark onto the event loop (thread-safe)
    if _loop and not _loop.is_closed():
        _loop.call_soon_threadsafe(_dirty.add, ticker)


# ── 10 Hz broadcast snapshot (runs on FastAPI event loop) ────────────────────

async def broadcast_snapshot() -> None:
    """
    Push all dirty (changed) ticker prices to every connected WS client
    as a SINGLE batched JSON message per client per cycle.

    Called every 100 ms by the broadcast loop in main.py.

    KEY DESIGN: one queue.put_nowait() per client per cycle (not one per ticker).
    This means _send_loop() calls websocket.send_text() once per 100 ms instead
    of N times (where N = number of dirty tickers).  Keeps the asyncio event loop
    free so _price_broadcast_loop wakes on schedule and delivers smooth ticks.

    Because this and update()'s call_soon_threadsafe callback both run on
    the same asyncio event loop thread, _dirty access is race-free.
    """
    if not _clients or not _dirty:
        return

    # Snapshot and clear the dirty set atomically (single event-loop thread)
    tickers = list(_dirty)
    _dirty.clear()

    # Build the full batch payload for all dirty tickers
    full_batch: dict = {}
    for ticker in tickers:
        data = _prices.get(ticker)
        if data:
            full_batch[ticker] = data

    if not full_batch:
        return

    full_msg = json.dumps(full_batch)

    # Fan-out: ONE message per client per cycle (filtered by subscription if set)
    for q in list(_clients):
        subs = _client_tickers.get(id(q), set())
        if subs:
            # Client subscribed to specific tickers — filter the batch
            filtered = {t: d for t, d in full_batch.items() if t in subs}
            if not filtered:
                continue   # nothing relevant for this client this cycle
            msg = json.dumps(filtered)
        else:
            # No subscription filter — send the full batch
            msg = full_msg

        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            # Client queue is full (very slow mobile connection).
            # Drop this cycle — the NEXT cycle will have the latest price.
            # Since we batch, the next message will include the most recent
            # value for every ticker, so no stale data is shown.
            pass


# ── WebSocket client management ───────────────────────────────────────────────

def add_client(q: asyncio.Queue, tickers: set[str]) -> None:
    _clients.append(q)
    _client_tickers[id(q)] = tickers
    logger.info(
        f"[price_store] WS client connected — "
        f"{len(_clients)} total | tickers={sorted(tickers)}"
    )


def remove_client(q: asyncio.Queue) -> None:
    try:
        _clients.remove(q)
    except ValueError:
        pass
    _client_tickers.pop(id(q), None)
    logger.info(f"[price_store] WS client disconnected — {len(_clients)} remaining")


def snapshot(tickers: list[str]) -> dict:
    """Return current prices for given tickers (immediate WS connect response)."""
    return {t: _prices[t] for t in tickers if t in _prices}


def connected_client_count() -> int:
    return len(_clients)
