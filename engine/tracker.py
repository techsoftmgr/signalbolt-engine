"""
Signal tracker — runs every 15 minutes alongside the scanner.

For every active signal with result='pending':
  1. Fetch current price via yfinance
  2. Check price against target_one, target_two, stop_loss
  3. On a hit: write result, hit_target, result_pct, result_pnl,
               closed_at, status='closed' back to Supabase

Required Supabase columns (run migration if not present):
  ALTER TABLE signals ADD COLUMN IF NOT EXISTS result      text    DEFAULT 'pending';
  ALTER TABLE signals ADD COLUMN IF NOT EXISTS hit_target  text;
  ALTER TABLE signals ADD COLUMN IF NOT EXISTS result_pct  float;
  ALTER TABLE signals ADD COLUMN IF NOT EXISTS result_pnl  float;
"""

import logging
import os
import sentry_sdk
from datetime import datetime, timezone, timedelta
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

logger = logging.getLogger("signalbolt.tracker")

# ── Alpaca client singleton ───────────────────────────────────
# Creating StockHistoricalDataClient is expensive (sets up connection pool).
# One module-level instance is shared across all _current_price() calls
# instead of re-creating per ticker per maintenance pass.
_alpaca_client = None
_alpaca_ok     = False

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    from engine.config import ALPACA_API_KEY, ALPACA_SECRET_KEY
    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        _alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        _alpaca_ok     = True
except Exception as _e:
    logger.debug(f"[tracker] Alpaca client init failed: {_e} — will use yfinance only")


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------

def _supabase_key() -> str:
    return os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]


def _supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], _supabase_key())


# ---------------------------------------------------------------------------
# Price fetch
# ---------------------------------------------------------------------------

def _current_price(ticker: str) -> Optional[float]:
    """
    Fetch latest trade price.
    Primary:  Alpaca REST latest trade (real-time SIP on paid plan).
              Uses module-level singleton client — no per-call reconnection.
    Fallback: yfinance fast_info (delayed, unofficial).
    """
    # ── Alpaca primary (reuses module-level singleton) ────────
    if _alpaca_ok and _alpaca_client is not None:
        try:
            trade = _alpaca_client.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=ticker)
            )
            price = float(trade[ticker].price)
            if price > 0:
                return price
        except Exception as e:
            logger.debug(f"[tracker] Alpaca price failed for {ticker}: {e}")

    # ── yfinance fallback ─────────────────────────────────────
    try:
        import yfinance as yf
        price = yf.Ticker(ticker).fast_info["last_price"]
        return float(price) if price else None
    except Exception as e:
        logger.warning(f"[tracker] price fetch failed for {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Level check
# ---------------------------------------------------------------------------

def _check_levels(signal: dict, price: float) -> Optional[dict]:
    """
    Returns a result dict if a level is hit, else None.
    """
    entry      = float(signal["entry_price"])
    stop       = float(signal["stop_loss"])
    target_one = float(signal["target_one"])
    target_two = float(signal["target_two"])
    is_long    = signal["direction"] == "LONG"

    hit_target = None
    result     = None

    if is_long:
        if price >= target_two:
            hit_target, result = "t2", "win"
        elif price >= target_one:
            hit_target, result = "t1", "win"
        elif price <= stop:
            hit_target, result = "sl", "loss"
    else:
        if price <= target_two:
            hit_target, result = "t2", "win"
        elif price <= target_one:
            hit_target, result = "t1", "win"
        elif price >= stop:
            hit_target, result = "sl", "loss"

    if result is None:
        return None

    # P&L — always positive for wins, negative for losses
    if is_long:
        pnl_pct = ((price - entry) / entry) * 100
        pnl_abs = price - entry
    else:
        pnl_pct = ((entry - price) / entry) * 100
        pnl_abs = entry - price

    return {
        "result":       result,
        "hit_target":   hit_target,
        "result_pct":   round(pnl_pct, 4),
        "result_pnl":   round(pnl_abs, 4),
        "status":       "closed",
        "closed_reason": "target_hit" if result == "win" else "stop_hit",
        "closed_at":    datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Expiry check (48h fallback)
# ---------------------------------------------------------------------------

def _is_expired(signal: dict) -> bool:
    try:
        created = datetime.fromisoformat(signal["created_at"].replace("Z", "+00:00"))
        return datetime.now(timezone.utc) - created > timedelta(hours=48)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main tracking loop
# ---------------------------------------------------------------------------

def _log_event(sb: Client, signal_id: str, outcome: dict) -> None:
    """Write a closed_win / closed_loss event to signal_events — best-effort."""
    try:
        result     = outcome["result"]
        hit        = outcome.get("hit_target", "").upper()
        pct        = outcome.get("result_pct", 0)
        event_type = "closed_win" if result == "win" else "closed_loss"
        hit_map    = {"T1": "Target 1", "T2": "Target 2", "SL": "Stop Loss"}
        hit_label  = hit_map.get(hit, hit)
        note = (
            f"{hit_label} hit — closed +{pct:.1f}%"
            if result == "win"
            else f"{hit_label} hit — stopped out {pct:.1f}%"
        )
        sb.table("signal_events").insert({
            "signal_id":  signal_id,
            "event_type": event_type,
            "note":       note,
        }).execute()
    except Exception as e:
        logger.debug(f"[tracker] event log failed: {e}")


def track_signals() -> None:
    logger.info("[tracker] Starting signal tracking pass")
    sb = _supabase()

    # Fetch active signals — filter result='pending' if column exists,
    # else fall back to all active signals
    try:
        rows = (
            sb.table("signals")
            .select("*")
            .eq("status", "active")
            .eq("result", "pending")
            .execute()
            .data
        )
    except Exception:
        # Column may not exist yet — fall back gracefully
        try:
            rows = (
                sb.table("signals")
                .select("*")
                .eq("status", "active")
                .execute()
                .data
            )
        except Exception as e:
            logger.error(f"[tracker] Failed to fetch signals: {e}")
            return

    if not rows:
        logger.info("[tracker] No active signals to track")
        return

    logger.info(f"[tracker] Checking {len(rows)} active signal(s)")
    wins = losses = expired = skipped = 0

    for sig in rows:
        ticker = sig["ticker"]

        # --- Expiry check first (no price fetch needed) ---
        if _is_expired(sig):
            try:
                sb.table("signals").update({
                    "status":        "closed",
                    "closed_reason": "expired",
                    "result":        "expired",
                    "closed_at":     datetime.now(timezone.utc).isoformat(),
                }).eq("id", sig["id"]).execute()
                logger.info(f"[tracker] EXPIRED  {ticker}")
                expired += 1
            except Exception as e:
                logger.error(f"[tracker] Failed to expire {ticker}: {e}")
            continue

        # --- Price fetch ---
        price = _current_price(ticker)
        if price is None:
            skipped += 1
            continue

        # --- Level check ---
        outcome = _check_levels(sig, price)
        if outcome is None:
            logger.debug(
                f"[tracker] {ticker} price={price:.2f} — no level hit "
                f"(entry={sig['entry_price']} t1={sig['target_one']} "
                f"t2={sig['target_two']} sl={sig['stop_loss']})"
            )
            skipped += 1
            continue

        # --- Write result ---
        try:
            sb.table("signals").update(outcome).eq("id", sig["id"]).execute()
            logger.info(
                f"[tracker] Signal closed: {ticker} result={outcome['result']} "
                f"hit={outcome['hit_target'].upper()} pnl={outcome['result_pct']:+.2f}%"
            )
            # Log to signal_events timeline
            _log_event(sb, sig["id"], outcome)
            if outcome["result"] == "win":
                wins += 1
            else:
                losses += 1
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(f"[tracker] Failed to update {ticker}: {e}")

    logger.info(
        f"[tracker] Pass complete - {wins}W / {losses}L / {expired} expired / {skipped} pending"
    )
