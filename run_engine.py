"""
Standalone scheduler — runs the signal engine every 15 minutes.
Usage: python run_engine.py
"""

import time
import logging
from datetime import datetime

from engine import smc, scorer, explainer
from main import run_ticker, save_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_engine")

TICKERS = [
    "AAPL", "TSLA", "NVDA", "SPY", "QQQ",
    "MSFT", "AMZN", "META", "GOOGL", "AMD",
]

INTERVAL_SECONDS = 15 * 60  # 15 minutes


def run_cycle():
    log.info(f"--- Starting cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    saved_count = 0
    for ticker in TICKERS:
        try:
            result = run_ticker(ticker)
            if result:
                saved_count += 1
        except Exception as e:
            log.error(f"{ticker}: unexpected error — {e}")
    log.info(f"--- Cycle complete: {saved_count}/{len(TICKERS)} signals saved ---\n")
    return saved_count


if __name__ == "__main__":
    log.info("SignalBolt engine starting…")
    log.info(f"Watching: {', '.join(TICKERS)}")
    log.info(f"Interval: {INTERVAL_SECONDS // 60} minutes")
    log.info("─" * 50)

    while True:
        try:
            run_cycle()
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break
        except Exception as e:
            log.error(f"Cycle error: {e}")

        log.info(f"Sleeping {INTERVAL_SECONDS // 60} minutes until next cycle…")
        time.sleep(INTERVAL_SECONDS)
