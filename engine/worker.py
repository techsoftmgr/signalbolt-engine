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


async def _seed_prices() -> None:
    """
    Best-effort REST snapshot seed so the WS has prices before first trade.

    Re-uses main._alpaca_stock_snapshots, which is the same helper the web
    process used before this split. We import lazily to avoid pulling FastAPI
    into the worker boot path until actually needed.
    """
    try:
        from engine import price_store
        from engine.runner import ALL_TICKERS
        from main import _alpaca_stock_snapshots  # type: ignore
        seed_tickers = list(dict.fromkeys(ALL_TICKERS))[:40]
        snaps = _alpaca_stock_snapshots(seed_tickers)
        for ticker, data in snaps.items():
            price_store.seed(
                ticker,
                data["price"],
                data["changePercent"],
                data.get("session", "market"),
            )
            chg = data["changePercent"]
            prev = data["price"] / (1 + chg / 100) if chg != -100 else data["price"]
            price_store.set_prev_close(ticker, prev)
        logger.info(f"[worker] Price store seeded with {len(snaps)} tickers")
    except Exception as e:
        logger.warning(f"[worker] Price seed failed (non-fatal): {e}")


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

    logger.info("SignalBolt worker started — scheduler + Alpaca stream + heartbeat active")

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
    import socket
    from datetime import datetime, timezone

    machine_id = _os.environ.get("FLY_MACHINE_ID", socket.gethostname())
    pid        = _os.getpid()
    disabled   = False

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
        if not disabled:
            try:
                await asyncio.to_thread(_write_heartbeat)
                logger.debug(f"[heartbeat] wrote beat for {HEARTBEAT_SERVICE_NAME}")
            except Exception as e:
                msg = str(e).lower()
                if "engine_heartbeats" in msg and ("does not exist" in msg or "schema cache" in msg):
                    logger.warning(
                        "[heartbeat] engine_heartbeats table missing — run "
                        "supabase-heartbeat-migration.sql to enable worker liveness alerts"
                    )
                    disabled = True
                else:
                    logger.warning(f"[heartbeat] write failed (non-fatal): {e}")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=HEARTBEAT_INTERVAL)
        except asyncio.TimeoutError:
            continue


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
