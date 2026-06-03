"""
Redis pub/sub bridge — receives ticks from the worker, feeds them into the
local price_store so /ws/prices can broadcast to connected phones.

Architecture after the worker split:
  signalbolt-worker:  Alpaca SIP → price_store.update() → publish_tick → Redis
  signalbolt-engine:  this subscriber → price_store.update_from_remote()
                      → broadcast_snapshot() → WS clients

Without this, the web's price_store stayed empty (worker had its own copy,
separate process) and the WS endpoint only ever delivered the initial REST
snapshot — users saw prices "freeze" within ~1 minute of opening the app.

Failure mode:
  - REDIS_URL unset → subscriber is a no-op (logged at startup once)
  - Redis disconnect → automatic reconnect with exponential backoff
  - Bad message payload → log warning, skip, keep going
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

from engine import price_store

logger = logging.getLogger("signalbolt.price_subscriber")

_REDIS_URL = os.environ.get("REDIS_URL", "").strip()
_RECONNECT_BACKOFF_INITIAL = 1.0   # seconds
_RECONNECT_BACKOFF_MAX     = 30.0


async def run_subscriber() -> None:
    """
    Long-running task: SUBSCRIBE to PRICE_CHANNEL forever, apply each tick.

    Designed to be spawned once from FastAPI lifespan startup. Survives
    Redis hiccups via reconnect-with-backoff; never re-raises so the
    surrounding app keeps running even if Redis is unavailable.
    """
    if not _REDIS_URL:
        logger.info("[price_subscriber] REDIS_URL not set — pub/sub bridge disabled "
                    "(WS clients will only receive ticks from in-process Alpaca stream, "
                    "which after the worker split is none)")
        return

    backoff = _RECONNECT_BACKOFF_INITIAL
    while True:
        client = None
        pubsub = None
        try:
            # redis-py 5.x ships an asyncio client out of the box
            import redis.asyncio as aioredis
            client = aioredis.from_url(
                _REDIS_URL,
                socket_timeout=10.0,
                socket_connect_timeout=5.0,
                health_check_interval=30,
            )
            pubsub = client.pubsub()
            await pubsub.subscribe(price_store.PRICE_CHANNEL)
            logger.info(f"[price_subscriber] Subscribed to {price_store.PRICE_CHANNEL} "
                        f"— relaying ticks to local price_store")
            backoff = _RECONNECT_BACKOFF_INITIAL   # reset on successful connect

            # Poll with a short timeout instead of a blocking listen(). Under
            # socket_timeout, a blocking read RAISES (tearing down the whole
            # connection) whenever the channel sits idle for >socket_timeout —
            # e.g. market closed and the worker publishes no ticks. That caused
            # a reconnect/resubscribe loop every ~10s. get_message(timeout=...)
            # uses a non-blocking readiness poll, so an idle interval just
            # returns None and we keep waiting. Calling get_message regularly
            # also lets redis-py fire its health_check_interval PINGs, which is
            # how a genuinely dead socket still gets detected.
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True,
                                               timeout=1.0)
                if msg is None:
                    continue   # idle this interval — no tick, keep waiting
                if msg.get("type") != "message":
                    continue   # 'subscribe' confirmations, pings, etc.
                try:
                    payload = json.loads(msg["data"])
                    # New batched format: {"ticks": {ticker: price, ...}}
                    # The worker now PUBLISHes at most 10×/sec with all dirty
                    # tickers bundled (see publish_batch_loop in price_store).
                    ticks = payload.get("ticks")
                    if isinstance(ticks, dict):
                        for ticker, val in ticks.items():
                            # New format: [price, changePercent]; old: bare price.
                            if isinstance(val, (list, tuple)):
                                price = val[0]
                                chg   = val[1] if len(val) > 1 else None
                            else:
                                price, chg = val, None
                            if ticker and price is not None:
                                price_store.update_from_remote(
                                    ticker, float(price),
                                    float(chg) if chg is not None else None,
                                )
                    else:
                        # Backward-compat for single-tick format (in case the
                        # worker is mid-deploy and still sending old format)
                        ticker = payload.get("t")
                        price  = payload.get("p")
                        if ticker and price is not None:
                            price_store.update_from_remote(ticker, float(price))
                except Exception as e:
                    # Don't let a single bad message kill the loop
                    logger.debug(f"[price_subscriber] bad message: {e}")

        except asyncio.CancelledError:
            # Clean shutdown (lifespan exit)
            logger.info("[price_subscriber] Cancelled — shutting down")
            raise
        except Exception as e:
            logger.warning(f"[price_subscriber] connection error: {e} — "
                           f"reconnecting in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX)
        finally:
            try:
                if pubsub is not None:
                    await pubsub.unsubscribe()
                    await pubsub.close()
            except Exception:
                pass
            try:
                if client is not None:
                    await client.aclose()
            except Exception:
                pass
