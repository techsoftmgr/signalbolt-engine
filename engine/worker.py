"""
SignalBolt worker process — runs the trading engine without FastAPI.

Responsibilities:
  • Initialise the in-memory price store
  • Seed prices from Alpaca REST snapshot
  • Start APScheduler (day_trade / swing / options_flow / dark_pool jobs)
  • Start the Alpaca WebSocket stream (real-time bars + trades)
  • Shut down all of the above cleanly on SIGTERM / SIGINT

This isolates the heavy/long-running work from the web process so:
  • /health stays fast (no contention)
  • the API VM can be scaled horizontally
  • the Alpaca SIP WebSocket runs from exactly ONE machine (worker count = 1)

Launched on Fly via the [processes].worker entry in fly.toml:
    worker = "python -m engine.worker"
"""

import asyncio
import logging
import signal
import sys
from contextlib import suppress

# UTF-8 stdout for emoji-safe logging on Windows (no-op on Linux/Fly)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("engine.worker")


async def _seed_prices(reseed: bool = False) -> None:
    """
    Best-effort REST snapshot seed so the WS has prices before first trade.

    Re-uses main._alpaca_stock_snapshots, which is the same helper the web
    process used before this split. We import lazily to avoid pulling FastAPI
    into the worker boot path until actually needed.

    reseed=True: refresh ONLY prev_close (the changePercent baseline) without
    clobbering the live price stream — used by the periodic refresh that keeps
    the baseline anchored to the current session's prior close. See
    price_store.reseed_prev_close for the why (stale-after-rollover bug).
    """
    try:
        from engine import price_store
        from engine.runner import ALL_TICKERS
        from main import _alpaca_stock_snapshots  # type: ignore
        seed_tickers = list(dict.fromkeys(ALL_TICKERS))[:40]
        snaps = _alpaca_stock_snapshots(seed_tickers)
        for ticker, data in snaps.items():
            chg  = data["changePercent"]
            # Derive the prior-day close Alpaca used (price / (1 + chg)). This
            # recovers previous_daily_bar.close exactly, which during RTH is the
            # CURRENT session's prior close (the correct baseline).
            prev = data["price"] / (1 + chg / 100) if chg != -100 else data["price"]
            if reseed:
                price_store.reseed_prev_close(ticker, prev)
            else:
                price_store.seed(
                    ticker,
                    data["price"],
                    chg,
                    data.get("session", "market"),
                )
                price_store.set_prev_close(ticker, prev)
        logger.info(
            f"[worker] Price store {'re-' if reseed else ''}seeded with {len(snaps)} tickers"
        )
    except Exception as e:
        logger.warning(f"[worker] Price {'re' if reseed else ''}seed failed (non-fatal): {e}")


async def _reseed_prev_close_loop(stop_event: asyncio.Event) -> None:
    """
    Periodically refresh prev_close so live changePercent stays anchored to the
    CURRENT session's prior close.

    The worker outlives every daily session rollover but seeds prev_close only
    once at boot, so without this the baseline freezes at boot-day's value and
    every streamed % drifts (live price stays correct, % is computed against a
    multi-day-old close — the watchlist "price right, % wrong" bug). Cheap: one
    batched Alpaca snapshot per cycle. Self-healing and never re-raises.
    """
    import os
    interval = int(os.environ.get("PREV_CLOSE_RESEED_SEC", "600"))  # 10 min
    logger.info(f"[worker] prev_close reseed loop started — every {interval}s")
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass   # interval elapsed → do a reseed
        except asyncio.CancelledError:
            logger.info("[worker] prev_close reseed loop cancelled")
            raise
        if stop_event.is_set():
            break
        try:
            await _seed_prices(reseed=True)
        except Exception as e:
            logger.warning(f"[worker] prev_close reseed cycle failed (non-fatal): {e}")


async def main() -> None:
    from engine.runner import start_scheduler
    from engine.stream import run_stream
    from engine import price_store

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_shutdown(*_args) -> None:
        logger.info("[worker] Shutdown signal received")
        stop_event.set()

    # signal handlers may not be supported on Windows event loops — guard.
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except (NotImplementedError, RuntimeError):
            # Fall back to default behaviour on platforms without async handlers
            try:
                signal.signal(sig, _request_shutdown)
            except Exception:
                pass

    # ── Boot ──────────────────────────────────────────────────────────────────
    price_store.init(loop)
    await _seed_prices()

    scheduler   = start_scheduler()
    stream_task = asyncio.create_task(run_stream(), name="alpaca_stream")
    heartbeat_task = asyncio.create_task(_heartbeat_loop(stop_event), name="heartbeat")
    # Publishes batched Alpaca ticks to Redis every 100ms so the web app
    # machines can broadcast them to their WS clients. No-op when
    # REDIS_URL isn't set. See engine/price_store.publish_batch_loop.
    publish_task = asyncio.create_task(
        price_store.publish_batch_loop(), name="price_publish_batch"
    )
    # Keeps the changePercent baseline anchored to the current session's prior
    # close (the worker is long-lived; boot-time prev_close goes stale after the
    # daily rollover → "price right, % wrong" watchlist bug).
    reseed_task = asyncio.create_task(
        _reseed_prev_close_loop(stop_event), name="prev_close_reseed"
    )

    logger.info("SignalBolt worker started — scheduler + Alpaca stream + heartbeat + Redis publish + prev_close reseed active")

    try:
        await stop_event.wait()
    finally:
        logger.info("[worker] Shutting down")

        # Stop the Alpaca WS first so the connection slot is released cleanly
        try:
            from engine.stream import _wss_ref
            if _wss_ref is not None:
                _wss_ref.stop()
                await asyncio.sleep(2)
        except Exception as e:
            logger.warning(f"[worker] Error stopping Alpaca WebSocket: {e}")

        stream_task.cancel()
        with suppress(asyncio.CancelledError):
            await stream_task

        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task

        publish_task.cancel()
        with suppress(asyncio.CancelledError):
            await publish_task

        reseed_task.cancel()
        with suppress(asyncio.CancelledError):
            await reseed_task

        try:
            scheduler.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"[worker] Scheduler shutdown error: {e}")

        logger.info("SignalBolt worker stopped cleanly")


# ── Heartbeat loop ────────────────────────────────────────────────────────────
# Worker writes a row to engine_heartbeats every HEARTBEAT_INTERVAL seconds.
# /ready on the app process reads that row; if last_beat is older than
# HEARTBEAT_STALE_AFTER seconds, /ready reports degraded so Sentry/uptime
# tooling can alert that the worker silently died (no Alpaca stream =
# no signals fire, no push notifications).
#
# The table is created by supabase-heartbeat-migration.sql. If the table
# doesn't exist yet the heartbeat is a no-op — it logs once and stops trying,
# so missing migrations don't crash the worker.

import os as _os

HEARTBEAT_INTERVAL     = int(_os.environ.get("WORKER_HEARTBEAT_INTERVAL_SEC", "60"))
HEARTBEAT_SERVICE_NAME = _os.environ.get("WORKER_SERVICE_NAME", "engine_worker")


async def _heartbeat_loop(stop_event: asyncio.Event) -> None:
    """
    Background loop: upsert engine_heartbeats every HEARTBEAT_INTERVAL seconds.

    Self-healing: if the table is missing (or any other error), we back off
    to HEARTBEAT_BACKOFF_SEC between attempts but never permanently disable.
    That way running the migration later, fixing the env, or transient
    Supabase errors all recover automatically without a worker restart.
    """
    import socket
    from datetime import datetime, timezone

    machine_id = _os.environ.get("FLY_MACHINE_ID", socket.gethostname())
    pid        = _os.getpid()
    # On consecutive failures, back off to this interval instead of normal cadence
    backoff_sec = int(_os.environ.get("HEARTBEAT_BACKOFF_SEC", "300"))
    last_was_failure = False

    def _write_heartbeat() -> None:
        # Sync call deliberately kept on its own thread to avoid blocking
        # the event loop. Runs every HEARTBEAT_INTERVAL seconds at most.
        from supabase import create_client
        sb_url = _os.environ["SUPABASE_URL"]
        sb_key = _os.environ.get("SUPABASE_KEY") or _os.environ["SUPABASE_SECRET_KEY"]
        sb     = create_client(sb_url, sb_key)
        sb.table("engine_heartbeats").upsert({
            "service":    HEARTBEAT_SERVICE_NAME,
            "last_beat":  datetime.now(timezone.utc).isoformat(),
            "pid":        pid,
            "machine_id": machine_id,
        }).execute()

    while not stop_event.is_set():
        try:
            await asyncio.to_thread(_write_heartbeat)
            if last_was_failure:
                logger.info(f"[heartbeat] recovered — writing beats for {HEARTBEAT_SERVICE_NAME}")
            else:
                logger.debug(f"[heartbeat] wrote beat for {HEARTBEAT_SERVICE_NAME}")
            last_was_failure = False
            sleep_for = HEARTBEAT_INTERVAL
        except Exception as e:
            msg = str(e).lower()
            if "engine_heartbeats" in msg and ("does not exist" in msg or "schema cache" in msg):
                logger.warning(
                    "[heartbeat] engine_heartbeats table missing — run "
                    "supabase-heartbeat-migration.sql. Will retry in %ss.",
                    backoff_sec,
                )
            else:
                logger.warning(f"[heartbeat] write failed (non-fatal): {e}. Will retry in {backoff_sec}s.")
            last_was_failure = True
            sleep_for = backoff_sec

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=sleep_for)
        except asyncio.TimeoutError:
            continue


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
