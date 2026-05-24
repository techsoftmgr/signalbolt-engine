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

    logger.info("SignalBolt worker started — scheduler + Alpaca stream active")

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

        try:
            scheduler.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"[worker] Scheduler shutdown error: {e}")

        logger.info("SignalBolt worker stopped cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
