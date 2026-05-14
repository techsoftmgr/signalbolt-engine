"""
Signal scanner — runs every 15 minutes via APScheduler.
For each ticker in WATCHLIST:
  1. Skip if an active signal already exists in Supabase
  2. Run SMC analysis (yfinance data)
  3. Score confluence across four layers
  4. If score >= 75: generate AI explanation and write signal to Supabase
"""

import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from supabase import create_client, Client

from engine import smc, scorer, explainer, options_scanner

load_dotenv()

logger = logging.getLogger(__name__)

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "TSLA", "AMD", "SPY", "QQQ", "COIN", "PLTR",
]

TIMEFRAME = "1h"


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def _supabase_key() -> str:
    return os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]


def _supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], _supabase_key())


def _has_active_signal(sb: Client, ticker: str) -> bool:
    try:
        result = (
            sb.table("signals")
            .select("id")
            .eq("ticker", ticker)
            .eq("status", "active")
            .execute()
        )
        return len(result.data) > 0
    except Exception as e:
        logger.error(f"[runner] Supabase active-signal check failed for {ticker}: {e}")
        return False


def _write_signal(sb: Client, row: dict) -> None:
    try:
        sb.table("signals").insert(row).execute()
        logger.info(
            f"[runner] SIGNAL SAVED  {row['ticker']:6s} {row['direction']:5s} "
            f"entry={row['entry_price']}  score={row['confidence_score']}"
        )
    except Exception as e:
        logger.error(f"[runner] Supabase insert failed for {row['ticker']}: {e}")


def _has_active_option_signal(sb: Client, ticker: str) -> bool:
    try:
        result = (
            sb.table("option_signals")
            .select("id")
            .eq("ticker", ticker)
            .eq("status", "active")
            .execute()
        )
        return len(result.data) > 0
    except Exception as e:
        logger.error(f"[runner] Option signal check failed for {ticker}: {e}")
        return False


def _write_option_signal(sb: Client, row: dict) -> None:
    try:
        sb.table("option_signals").insert(row).execute()
        logger.info(
            f"[runner] OPTION SAVED  {row['ticker']:6s} {row['contract_type']:4s} "
            f"strike={row['strike_price']}  exp={row['expiry_date']}  score={row['confidence_score']}"
        )
    except Exception as e:
        logger.error(f"[runner] Option signal insert failed for {row['ticker']}: {e}")


# ---------------------------------------------------------------------------
# Per-ticker pipeline
# ---------------------------------------------------------------------------

def _process_ticker(sb: Client, ticker: str) -> None:
    # 1. Skip if already active
    if _has_active_signal(sb, ticker):
        logger.debug(f"[runner] {ticker}: active signal exists — skipping")
        return

    # 2. SMC analysis
    analysis = smc.analyze(ticker, interval=TIMEFRAME)
    if not analysis or not analysis.get("direction"):
        logger.info(f"[runner] {ticker}: no clear SMC direction")
        return

    # 3. Confluence score
    scored = scorer.score(analysis)
    logger.info(
        f"[runner] {ticker}: score={scored['total']} "
        f"(L1={scored['breakdown']['l1_smc']} "
        f"L2={scored['breakdown']['l2_technical']} "
        f"L3={scored['breakdown']['l3_sentiment']} "
        f"L4={scored['breakdown']['l4_risk']} "
        f"L5={scored['breakdown'].get('l5_mtf', '—')})"
    )

    if not scored["passes"]:
        from engine.scorer import FIRE_THRESHOLD
        logger.info(f"[runner] {ticker}: score {scored['total']} < {FIRE_THRESHOLD} — not firing")
        return

    # 4. Build signal row
    signal_row = {
        "ticker":           ticker,
        "direction":        scored["direction"],
        "entry_price":      scored["entry"],
        "stop_loss":        scored["stop_loss"],
        "target_one":       scored["target_one"],
        "target_two":       scored["target_two"],
        "confidence_score": scored["total"],
        "timeframe":        TIMEFRAME,
        "status":           "active",
        "ai_explanation":   None,
    }

    # 5. Generate AI explanation
    signal_row["ai_explanation"] = explainer.generate(signal_row, scored["breakdown"])

    # 6. Persist stock signal
    _write_signal(sb, signal_row)

    # 7. Scan options chain and persist option signal if contract found
    if not _has_active_option_signal(sb, ticker):
        opt = options_scanner.scan(
            ticker, scored["direction"],
            analysis["current_price"],
            stock_target_one=scored.get("target_one"),
        )
        if opt:
            opt["confidence_score"] = scored["total"]
            opt["ai_explanation"]   = signal_row["ai_explanation"]
            opt["timeframe"]        = TIMEFRAME
            opt["status"]           = "active"
            _write_option_signal(sb, opt)
        else:
            logger.debug(f"[runner] {ticker}: no options contract found")


# ---------------------------------------------------------------------------
# Auto-close logic
# ---------------------------------------------------------------------------

def _close_signals(sb: Client) -> None:
    """
    Check every active signal and close it if:
      - target_two hit   → closed_reason = 'target_hit'
      - stop_loss hit    → closed_reason = 'stop_hit'
      - older than 48h   → closed_reason = 'expired'
    """
    import yfinance as yf
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=48)

    # ── Stock signals ──
    try:
        rows = sb.table("signals").select("*").eq("status", "active").execute().data
    except Exception as e:
        logger.error(f"[closer] fetch signals failed: {e}")
        rows = []

    for sig in rows:
        created = datetime.fromisoformat(sig["created_at"].replace("Z", "+00:00"))
        reason  = None

        # Expiry check first (no price fetch needed)
        if created < cutoff:
            reason = "expired"
        else:
            try:
                price = yf.Ticker(sig["ticker"]).fast_info.last_price
                if price:
                    if sig["direction"] == "LONG":
                        if price >= sig["target_two"]:  reason = "target_hit"
                        elif price <= sig["stop_loss"]: reason = "stop_hit"
                    else:
                        if price <= sig["target_two"]:  reason = "target_hit"
                        elif price >= sig["stop_loss"]: reason = "stop_hit"
            except Exception:
                pass

        if reason:
            try:
                sb.table("signals").update({
                    "status":        "closed",
                    "closed_reason": reason,
                    "closed_at":     now.isoformat(),
                }).eq("id", sig["id"]).execute()
                logger.info(f"[closer] CLOSED stock {sig['ticker']} ({reason})")
            except Exception as e:
                logger.error(f"[closer] close failed for {sig['ticker']}: {e}")

    # ── Option signals ──
    try:
        opt_rows = sb.table("option_signals").select("*").eq("status", "active").execute().data
    except Exception as e:
        logger.error(f"[closer] fetch option_signals failed: {e}")
        opt_rows = []

    for sig in opt_rows:
        created = datetime.fromisoformat(sig["created_at"].replace("Z", "+00:00"))
        reason  = None

        if created < cutoff:
            reason = "expired"
        else:
            try:
                price      = yf.Ticker(sig["ticker"]).fast_info.last_price
                delta      = sig.get("delta") or 0
                und_price  = sig.get("underlying_price") or 0
                entry_prem = sig.get("entry_premium") or 0
                if price and und_price and delta:
                    est = entry_prem + delta * (price - und_price)
                    if est >= sig["target_premium"]: reason = "target_hit"
                    elif est <= sig["stop_premium"]: reason = "stop_hit"
            except Exception:
                pass

        if reason:
            try:
                sb.table("option_signals").update({
                    "status":        "closed",
                    "closed_reason": reason,
                    "closed_at":     now.isoformat(),
                }).eq("id", sig["id"]).execute()
                logger.info(f"[closer] CLOSED option {sig['ticker']} {sig['contract_type']} ({reason})")
            except Exception as e:
                logger.error(f"[closer] option close failed for {sig['ticker']}: {e}")


# ---------------------------------------------------------------------------
# Full scan
# ---------------------------------------------------------------------------

def run_scan(tickers: Optional[List[str]] = None) -> None:
    watchlist = tickers or WATCHLIST
    logger.info(
        f"[runner] Scan started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} "
        f"— {len(watchlist)} tickers"
    )
    sb = _supabase()

    # Close any signals that hit target/stop/expiry before scanning for new ones
    _close_signals(sb)

    fired = 0
    for ticker in watchlist:
        try:
            _process_ticker(sb, ticker)
            fired += 1
        except Exception as e:
            logger.error(f"[runner] Unexpected error for {ticker}: {e}", exc_info=True)

    logger.info(f"[runner] Scan complete — processed {fired}/{len(watchlist)} tickers")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def start_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        run_scan,
        trigger=IntervalTrigger(minutes=15),
        id="signal_scan",
        name="SignalBolt 15-min scan",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc),  # run immediately on first start
    )
    scheduler.start()
    logger.info("[runner] Scheduler started — every 15 minutes")
    return scheduler
