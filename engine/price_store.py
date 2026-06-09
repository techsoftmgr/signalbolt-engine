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
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger("signalbolt.price_store")

ET = ZoneInfo("America/New_York")

# ── Redis pub/sub bridge ──────────────────────────────────────────────────────
# The worker process owns the Alpaca SIP subscription; the web process serves
# /ws/prices to phones. They live in separate Fly apps after the worker split,
# so they no longer share in-process state. This module-level pub/sub bridge
# fans price updates from worker → all web machines via a Redis channel.
#
# Worker side: each tick calls publish_tick() → PUBLISH to PRICE_CHANNEL
# Web side:    price_subscriber.py SUBSCRIBE → calls update_from_remote()
#              → existing broadcast_snapshot() pushes to WS clients
#
# Why "signalbolt:price-ticks": namespaced so we don't collide if Redis is
# shared with other apps later.

PRICE_CHANNEL = "signalbolt:price-ticks"
_REDIS_URL = os.environ.get("REDIS_URL", "").strip()

# Batched publish — accumulates the latest price per ticker between flushes.
# publish_batch_loop() runs every PUBLISH_INTERVAL_SEC and emits ONE Redis
# message with all dirty tickers. Why batch:
#   - Per-tick PUBLISH at peak (300 trades/sec) = ~28M ops/month on Upstash
#     pay-as-you-go = ~$56/month, growing with ticker count
#   - Batched at 10Hz = ~1M ops/month = ~$2/month, regardless of how busy
#     individual tickers are
#   - End-user UX is identical: the web's broadcast_snapshot also runs at
#     10Hz, so publishing faster than that is wasted bandwidth — phones
#     can't render updates faster than the WS frame cadence anyway
PUBLISH_INTERVAL_SEC = 0.1   # 100ms = 10Hz, matches the WS broadcast cadence

# Latest price per ticker since the last flush. Written by update() (Alpaca
# SDK thread), read+cleared by publish_batch_loop() (asyncio thread). dict
# operations are GIL-protected for single get/set — no lock needed because
# the worst race is overwriting an old price with a newer one, which is
# exactly what we want.
_publish_buffer: dict[str, float] = {}

# Lazy-connected redis-py client (sync). One pooled connection per worker
# process is plenty — we PUBLISH at most 10 times per second now.
_publisher = None
_publisher_failed_at: float = 0.0
_PUBLISHER_RETRY_COOLDOWN_SEC = 30  # don't hammer a dead Redis


def _get_publisher():
    """Lazy redis client for PUBLISH. Returns None if Redis isn't configured."""
    global _publisher
    if not _REDIS_URL:
        return None
    if _publisher is None:
        import redis as _redis
        _publisher = _redis.Redis.from_url(
            _REDIS_URL,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
            retry_on_timeout=False,
        )
    return _publisher


def publish_tick(ticker: str, price: float) -> None:
    """
    Queue a tick for the next batched Redis PUBLISH cycle.

    Called from update() (Alpaca SDK thread). Just sets a dict entry —
    no I/O, no event-loop interaction, no chance of blocking. The actual
    network publish happens at most 10×/sec inside publish_batch_loop()
    on the worker's asyncio thread.

    If two trades arrive between flushes, only the latest price is kept
    (intentional — phones can't see intermediate prices at 10Hz anyway,
    so why pay Redis to ship them).
    """
    if not _REDIS_URL:
        return
    # Carry the worker's freshly-computed changePercent alongside the price, so the
    # web doesn't have to recompute it from its own (often un-seeded) prev_close.
    _publish_buffer[ticker] = (round(price, 4), (_prices.get(ticker) or {}).get("changePercent"))


async def publish_batch_loop() -> None:
    """
    Spawn this from worker.main() as a background asyncio task.

    Every PUBLISH_INTERVAL_SEC, flushes _publish_buffer into a single Redis
    PUBLISH message:
        {"ticks": {"NVDA": 1234.56, "AAPL": 180.25, ...}}

    Idle cycles (empty buffer) cost zero Redis ops — just a sleep and a
    set check.

    Reconnect / cooldown handled inline; never re-raises so this task
    stays alive for the worker's lifetime.
    """
    global _publisher_failed_at
    import time as _time
    if not _REDIS_URL:
        logger.info("[price_store] REDIS_URL not set — publish batch loop disabled")
        return

    logger.info(f"[price_store] Publish batch loop started — flushing every "
                f"{int(PUBLISH_INTERVAL_SEC * 1000)}ms")

    while True:
        try:
            await asyncio.sleep(PUBLISH_INTERVAL_SEC)
        except asyncio.CancelledError:
            logger.info("[price_store] Publish batch loop cancelled")
            raise

        if not _publish_buffer:
            continue

        # In cooldown after a recent failure? Skip this cycle, but DON'T
        # clear the buffer — next successful cycle will pick up the latest
        # prices anyway (we only store latest per ticker).
        if _publisher_failed_at and (_time.monotonic() - _publisher_failed_at) < _PUBLISHER_RETRY_COOLDOWN_SEC:
            continue

        # Snapshot + clear buffer atomically by reassigning the module-level
        # dict. Any concurrent update() writes that arrive AFTER this line
        # go into the new empty dict and get picked up next cycle.
        batch = _publish_buffer.copy()
        _publish_buffer.clear()

        try:
            client = _get_publisher()
            if client:
                # Compact payload: {"ticks": {ticker: price, ...}}
                # ~10-15 bytes per ticker. A 50-ticker batch = ~700 bytes.
                client.publish(PRICE_CHANNEL, json.dumps({"ticks": batch}))
        except Exception as e:
            _publisher_failed_at = _time.monotonic()
            # Buffer was already cleared above; we lose this batch but the
            # next tick from each ticker will re-populate. Not worth holding
            # a queue of stale prices — phones want LATEST, not history.
            logger.warning(f"[price_store] Redis publish failed (batch={len(batch)}): {e}")

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
    # Seed prev_close too — without this, live ticks for this ticker fell to the
    # "prev unknown → 0%" branch and broadcast changePercent=0, clobbering this
    # correct REST value on the client (the HOOD "+0.00%" bug).
    if price > 0 and change_pct is not None and change_pct != -100:
        _prev_close[ticker] = price / (1 + change_pct / 100)


# ── Session helper ────────────────────────────────────────────────────────────

def _market_session_now() -> str:
    now = datetime.now(ET)
    m   = now.hour * 60 + now.minute
    wd  = now.weekday()   # Mon=0 .. Sun=6
    # Overnight (Blue Ocean) ~8pm-4am ET, Sun night → Fri morning. A tick in this
    # window is the overnight session, so it's tagged 'overnight' for the UI chip
    # (display-only; the signal engine never streams here).
    if m >= 1200:                          # 8:00 PM+
        return "overnight" if wd in (6, 0, 1, 2, 3) else "closed"
    if m < 240:                            # before 4:00 AM
        return "overnight" if wd in (0, 1, 2, 3, 4) else "closed"
    if   m < 570:   return "pre"      # 4:00–9:29 AM ET
    elif m < 960:   return "market"   # 9:30 AM–3:59 PM ET
    else:            return "post"     # 4:00–7:59 PM ET


# ── Called from Alpaca trade stream ──────────────────────────────────────────

def update(ticker: str, trade_price: float) -> None:
    """
    Record a live trade price and mark the ticker dirty for the next
    broadcast cycle.  Called from the Alpaca SDK thread (worker process)
    — must not touch asyncio primitives directly.  Uses call_soon_threadsafe
    to queue the dirty-set mutation onto the FastAPI event loop.

    Also publishes the tick to Redis so other engine machines (the web
    app(s) after the worker split) can broadcast it to their own WS
    clients. Without that bridge, web's price_store stays empty and
    /ws/prices delivers only the initial snapshot, no live ticks.
    """
    _update_local(ticker, trade_price)
    # Fan out to other machines via Redis. No-op if REDIS_URL unset.
    publish_tick(ticker, trade_price)


def update_from_remote(ticker: str, trade_price: float, change_pct: float | None = None) -> None:
    """
    Apply a tick that arrived via Redis pub/sub from the worker.
    Same as update() but does NOT re-publish (avoids a loop where every
    web machine re-broadcasts every tick it receives back to Redis).
    `change_pct` is the worker's authoritative value (it owns prev_close).
    """
    _update_local(ticker, trade_price, remote_chg=change_pct)


def _update_local(ticker: str, trade_price: float, remote_chg: float | None = None) -> None:
    """Shared logic for both local trade updates and remote (pub/sub) updates."""
    entry: dict = {"price": round(trade_price, 2), "session": _market_session_now()}
    prev = _prev_close.get(ticker)
    if remote_chg is not None:
        # Trust the worker's changePercent (it owns prev_close for streamed names).
        entry["changePercent"] = round(remote_chg, 3)
    elif prev:
        entry["changePercent"] = round((trade_price - prev) / prev * 100, 3)
    else:
        # prev_close unknown — do NOT broadcast a misleading 0% (it would clobber
        # the correct REST-seeded value on the client). Keep the last known value
        # if any; otherwise omit changePercent so the client retains its /prices %.
        last = (_prices.get(ticker) or {}).get("changePercent")
        if last is not None:
            entry["changePercent"] = last
    _prices[ticker] = entry

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
