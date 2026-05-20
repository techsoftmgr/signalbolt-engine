"""
Signal Monitor
==============
Runs every 15 minutes alongside tracker.py. Handles everything tracker doesn't:

  1. Market-close enforcer (3:50 PM ET)
     - day_trade + options_flow signals → force close at 3:50 PM ET
     - Scalping signals → force close after 30 min regardless of price
     - Swing trade → not affected (held overnight by design)
     - Push: "Market closing — close your [TICKER] position now"

  2. Structure reversal detector (stock signals only)
     - Re-runs SMC structure check on every active signal's ticker
     - LONG signal + bearish CHoCH detected → early exit
     - SHORT signal + bullish CHoCH detected → early exit
     - Push: "Structure reversed on [TICKER] — consider closing"

  3. T1 hit → SL moves to breakeven
     - When price crosses T1 but T2 not yet hit
     - Updates stop_loss = entry_price in DB so any reversal is a scratch, not a loss
     - Push: "[TICKER] hit T1 — stop moved to breakeven, riding to T2"

  4. Close notifications for ALL events
     - Target hit, stop hit, market close, reversal — all get push notifications

This module is intentionally separate from tracker.py so either can be
upgraded independently.
"""

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from supabase import create_client, Client

from engine import smc, push
from engine import alpaca_client as _alpaca

logger = logging.getLogger("signalbolt.monitor")

# ── In-memory status cache ────────────────────────────────────────────────────
# Tracks last known status per signal_id so we only log events on transitions.
# Lost on engine restart — that's fine, the signal_events table is the truth.
# Structure: { signal_id: "near_stop" | "below_entry" | "in_profit" |
#                         "building_profit" | "strong_profit" | "at_target" }
_STATUS_CACHE: dict[str, str] = {}
# Track which signals have already received an EOD warning this session
_EOD_WARNED: set[str] = set()

ET = ZoneInfo("America/New_York")

# ── Strategy close rules ──────────────────────────────────────────────────────
# Strategies that must be closed by market close (not held overnight)
INTRADAY_STRATEGIES = {"scalping", "day_trade", "options_flow"}

# Scalping max hold in minutes (regardless of market hours)
SCALP_MAX_HOLD_MINS = 30

# Market close stages for intraday signals
# EOD_WARN  → push "N min to close, consider booking profit" (no auto-close)
# FORCE_CLOSE → auto-close all intraday positions with accurate P&L
EOD_WARN_HOUR,   EOD_WARN_MINUTE   = 15, 0    # 3:00 PM ET — early profit warning
MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE = 15, 30  # 3:30 PM ET — force-close all intraday


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _supabase() -> Client:
    key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
    return create_client(os.environ["SUPABASE_URL"], key)


def _now_et() -> datetime:
    return datetime.now(ET)


def _is_market_hours() -> bool:
    """True during regular US market hours (9:30 AM – 4:00 PM ET, Mon-Fri)."""
    now = _now_et()
    if now.weekday() >= 5:      # Saturday/Sunday
        return False
    open_time  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_time <= now <= close_time


def _is_near_market_close() -> bool:
    """True from 3:30 PM ET — force-close window for all intraday positions."""
    now = _now_et()
    if now.weekday() >= 5:
        return False
    trigger = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    hard_close = now.replace(hour=16, minute=5, second=0, microsecond=0)
    return trigger <= now <= hard_close


def _is_eod_warning() -> bool:
    """True from 3:00 PM until force-close kicks in at 3:30 PM."""
    now = _now_et()
    if now.weekday() >= 5:
        return False
    warn_start  = now.replace(hour=EOD_WARN_HOUR,    minute=EOD_WARN_MINUTE,    second=0, microsecond=0)
    force_start = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    return warn_start <= now < force_start


def _current_price(ticker: str) -> Optional[float]:
    """
    Get current price via Alpaca (real-time, SIP feed) with yfinance fallback.
    Uses the shared singleton client — no per-call reconnection overhead.
    yfinance fast_info is 15-min delayed — Alpaca is real-time SIP.
    """
    # ── Alpaca primary (real-time SIP, shared singleton) ─────────────────────
    price = _alpaca.get_latest_price(ticker)
    if price:
        return price

    # ── yfinance fallback (15-min delayed, better than nothing) ──────────────
    try:
        import yfinance as yf
        p = yf.Ticker(ticker).fast_info["last_price"]
        return float(p) if p else None
    except Exception:
        return None


# ── Status derivation ─────────────────────────────────────────────────────────

def _derive_status(price: float, entry: float, t1: float, sl: float,
                   direction: str) -> str:
    """
    Compute a status label from price vs key levels.
    Returns one of: near_stop | below_entry | in_profit |
                    building_profit | strong_profit | at_target
    """
    is_long    = direction == "LONG"
    stop_dist  = abs(entry - sl)
    target_dist = abs(t1 - entry)

    if target_dist == 0:
        return "in_profit" if (is_long and price > entry) or (not is_long and price < entry) else "below_entry"

    # Progress toward target (negative = going wrong way)
    progress = ((price - entry) / target_dist) if is_long else ((entry - price) / target_dist)

    near_stop_pct = ((entry - price) / stop_dist) if is_long else ((price - entry) / stop_dist)

    if near_stop_pct >= 0.80:          return "near_stop"
    if progress >= 1.0:                return "at_target"
    if progress >= 0.60:               return "strong_profit"
    if progress >= 0.30:               return "building_profit"
    if progress > 0:                   return "in_profit"
    return "below_entry"


# ── Momentum analysis for early booking ──────────────────────────────────────

def _momentum_check(ticker: str, direction: str) -> tuple[bool, str]:
    """
    Returns (book_now, reason) using RSI + MACD on 5-min bars.
    book_now=True means momentum is failing — engine recommends booking profit.
    Uses Alpaca real-time bars (primary) with yfinance fallback.
    """
    try:
        import ta

        # ── Alpaca primary — real-time 5-min bars ─────────────────────────────
        close = None
        df_alpaca = _alpaca.get_bars(ticker, timeframe="5Min", days=2)
        if df_alpaca is not None and len(df_alpaca) >= 20:
            close = df_alpaca["close"]
        else:
            # ── yfinance fallback (15-min delayed) ────────────────────────────
            import yfinance as yf
            df_yf = yf.download(ticker, period="2d", interval="5m",
                                progress=False, auto_adjust=True)
            if not df_yf.empty and len(df_yf) >= 20:
                close = df_yf["Close"].squeeze()

        if close is None:
            return False, ""

        rsi         = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        macd        = ta.trend.MACD(close)
        macd_line   = macd.macd().iloc[-1]
        signal_line = macd.macd_signal().iloc[-1]

        is_long = direction == "LONG"

        # Overbought / oversold
        if is_long and rsi > 72:
            return True, f"RSI overbought ({rsi:.0f}) — momentum likely to stall"
        if not is_long and rsi < 28:
            return True, f"RSI oversold ({rsi:.0f}) — momentum likely to stall"

        # MACD bearish crossover (for LONG) or bullish (for SHORT)
        if is_long and macd_line < signal_line and macd_line > 0:
            return True, "MACD bearish crossover — momentum weakening near target"
        if not is_long and macd_line > signal_line and macd_line < 0:
            return True, "MACD bullish crossover — momentum weakening near target"

    except Exception as e:
        logger.debug(f"[monitor] Momentum check failed for {ticker}: {e}")

    return False, ""


# ── Status event helpers ──────────────────────────────────────────────────────

_STATUS_LABELS = {
    "near_stop":       ("⚠️", "Near Stop — Watch closely"),
    "below_entry":     ("↩",  "Below Entry — Waiting for recovery"),
    "in_profit":       ("💹", "In Profit — Hold, target not reached"),
    "building_profit": ("📈", "Building Profit — Approaching target"),
    "strong_profit":   ("🔥", "Strong Profit — Consider booking partial"),
    "at_target":       ("🎯", "At Target — Book profit"),
}


def _log_status_event(sb: Client, sig_id: str, status: str,
                      price: float | None, extra: str = "") -> None:
    """Log a status-change event to signal_events timeline."""
    emoji, base_label = _STATUS_LABELS.get(status, ("•", status))
    note = f"{emoji} {base_label}"
    if extra:
        note += f" — {extra}"
    if price:
        note += f" (${price:.2f})"
    _log_event(sb, sig_id, status, price=price, note=note)


def _close_signal(
    sb: Client,
    sig_id: str,
    reason: str,
    close_type: str = "stock",
    current_price: float | None = None,
    entry_price: float | None = None,
    direction: str = "LONG",
) -> None:
    """
    Write closed status to Supabase.
    When current_price + entry_price are provided (e.g. market_close, time_limit)
    the actual P&L is recorded so history shows win/loss — not just 'expired'.
    """
    table = "option_signals" if close_type == "option" else "signals"

    result    = "expired"
    pnl_pct   = None
    pnl_abs   = None

    if current_price and entry_price and entry_price > 0:
        is_long = direction == "LONG"
        raw_pct = ((current_price - entry_price) / entry_price * 100)
        pnl_pct = raw_pct if is_long else -raw_pct
        pnl_abs = (current_price - entry_price) if is_long else (entry_price - current_price)
        result  = "win" if pnl_pct > 0 else "loss"

    payload: dict = {
        "status":        "closed",
        "closed_reason": reason,
        "result":        result,
        "closed_at":     datetime.now(timezone.utc).isoformat(),
    }
    if pnl_pct is not None:
        payload["result_pct"] = round(pnl_pct, 4)
        payload["result_pnl"] = round(pnl_abs, 4)

    try:
        sb.table(table).update(payload).eq("id", sig_id).execute()
    except Exception as e:
        logger.error(f"[monitor] Close failed for {sig_id}: {e}")


def _update_sl(sb: Client, sig_id: str, new_sl: float) -> None:
    """Move stop loss to new level (e.g. breakeven after T1 hit)."""
    try:
        sb.table("signals").update({
            "stop_loss": round(new_sl, 4),
        }).eq("id", sig_id).execute()
    except Exception as e:
        logger.error(f"[monitor] SL update failed for {sig_id}: {e}")


def _log_event(
    sb: Client,
    signal_id: str,
    event_type: str,
    price: float | None = None,
    note: str = "",
) -> None:
    """Insert a row into signal_events — best-effort, never raises."""
    try:
        sb.table("signal_events").insert({
            "signal_id":  signal_id,
            "event_type": event_type,
            "price":      price,
            "note":       note,
        }).execute()
    except Exception as e:
        logger.debug(f"[monitor] event log failed ({event_type}): {e}")


# ---------------------------------------------------------------------------
# Reversal detection
# ---------------------------------------------------------------------------

def _detect_structure_reversal(ticker: str, direction: str) -> bool:
    """
    Returns True if SMC structure has flipped against the open signal direction.
    LONG signal + bearish CHoCH → reversal detected.
    SHORT signal + bullish CHoCH → reversal detected.
    """
    try:
        df = smc.fetch_candles(ticker, period="2d", interval="15m")
        if df.empty or len(df) < 20:
            return False
        df     = smc.detect_swings(df)
        struct = smc.detect_structure(df)

        if direction == "LONG" and struct.get("choch_bearish"):
            logger.info(f"[monitor] {ticker} LONG — bearish CHoCH detected → reversal")
            return True
        if direction == "SHORT" and struct.get("choch_bullish"):
            logger.info(f"[monitor] {ticker} SHORT — bullish CHoCH detected → reversal")
            return True
        return False
    except Exception as e:
        logger.debug(f"[monitor] Reversal check failed for {ticker}: {e}")
        return False


# ---------------------------------------------------------------------------
# Push helpers
# ---------------------------------------------------------------------------

def _push_market_close(ticker: str, direction: str, strategy: str, pnl_pct: float | None = None) -> None:
    if pnl_pct is not None and pnl_pct > 0:
        title = f"✅ {ticker} closed +{pnl_pct:.1f}% — Market Close"
        body  = f"Booked profit on {direction} {strategy.replace('_',' ')} before 4 PM ET"
    elif pnl_pct is not None and pnl_pct <= 0:
        title = f"⏰ {ticker} closed {pnl_pct:.1f}% — Market Close"
        body  = f"Position exited to avoid overnight risk. {direction} {strategy.replace('_',' ')}"
    else:
        title = f"⏰ Market Closing — {ticker}"
        body  = f"Close your {direction} {strategy.replace('_',' ')} position before 4 PM ET"
    push._send_raw(
        title=title, body=body,
        data={"type": "market_close", "ticker": ticker},
    )


def _push_eod_warning(ticker: str, direction: str, pnl_pct: float) -> None:
    now_et   = datetime.now(ZoneInfo("America/New_York"))
    mins_left = max(0, (16 * 60) - (now_et.hour * 60 + now_et.minute))
    push._send_raw(
        title=f"📊 {ticker} +{pnl_pct:.1f}% — {mins_left} min to close",
        body=f"Your {direction} signal is in profit. Consider booking before market close.",
        data={"type": "eod_warning", "ticker": ticker},
    )


def _push_early_book(ticker: str, direction: str, pnl_pct: float, reason: str) -> None:
    push._send_raw(
        title=f"💡 Book Profit Now — {ticker} +{pnl_pct:.1f}%",
        body=reason,
        data={"type": "book_profit", "ticker": ticker},
    )


def _push_status_change(ticker: str, status: str, pnl_pct: float | None) -> None:
    emoji, label = _STATUS_LABELS.get(status, ("•", status))
    pnl_str = f" ({'+' if pnl_pct and pnl_pct > 0 else ''}{pnl_pct:.1f}%)" if pnl_pct is not None else ""
    push._send_raw(
        title=f"{emoji} {ticker}{pnl_str} — {label.split(' — ')[0]}",
        body=label.split(" — ", 1)[-1] if " — " in label else label,
        data={"type": "status_change", "ticker": ticker, "status": status},
    )


def _push_reversal(ticker: str, direction: str) -> None:
    opposite = "BEARISH" if direction == "LONG" else "BULLISH"
    push._send_raw(
        title=f"⚠ Structure Reversed — {ticker}",
        body=f"{opposite} CHoCH detected. Consider closing your {direction} position.",
        data={"type": "reversal", "ticker": ticker, "direction": direction},
    )


def _push_t1_breakeven(ticker: str, direction: str, pct: float) -> None:
    push._send_raw(
        title=f"🎯 T1 Hit — {ticker} +{pct:.1f}%",
        body=f"Stop moved to breakeven. Riding to T2. {direction} still open.",
        data={"type": "t1_breakeven", "ticker": ticker},
    )


def _push_scalp_expired(ticker: str, direction: str) -> None:
    push._send_raw(
        title=f"⏱ Scalp Time Limit — {ticker}",
        body=f"30-min scalp window closed. Exit your {direction} position now.",
        data={"type": "scalp_expired", "ticker": ticker},
    )


def _push_closed(ticker: str, direction: str, result: str, pct: float) -> None:
    if result == "win":
        push._send_raw(
            title=f"✅ Target Hit — {ticker} +{pct:.1f}%",
            body=f"{direction} signal closed with a win.",
            data={"type": "signal_closed", "result": "win", "ticker": ticker},
        )
    elif result == "loss":
        push._send_raw(
            title=f"🔴 Stop Hit — {ticker} {pct:.1f}%",
            body=f"{direction} signal stopped out.",
            data={"type": "signal_closed", "result": "loss", "ticker": ticker},
        )


# ---------------------------------------------------------------------------
# Stock signal monitor
# ---------------------------------------------------------------------------

def _monitor_stocks(sb: Client) -> None:
    try:
        rows = (
            sb.table("signals")
            .select("*")
            .eq("status", "active")
            .execute()
            .data
        ) or []
    except Exception as e:
        logger.error(f"[monitor] Failed to fetch stock signals: {e}")
        return

    if not rows:
        return

    logger.info(f"[monitor] Checking {len(rows)} active stock signal(s)")
    near_close  = _is_near_market_close()
    eod_warning = _is_eod_warning()
    market_open = _is_market_hours()
    now_utc     = datetime.now(timezone.utc)

    for sig in rows:
        ticker    = sig["ticker"]
        strategy  = sig.get("strategy_type") or "day_trade"
        direction = sig["direction"]
        created   = datetime.fromisoformat(sig["created_at"].replace("Z", "+00:00"))
        age_mins  = (now_utc - created).total_seconds() / 60

        # ── 1. Scalp time limit (30 min, any time) ───────────────────────────
        if strategy == "scalping" and age_mins >= SCALP_MAX_HOLD_MINS:
            price = _current_price(ticker)
            logger.info(f"[monitor] {ticker} scalp time limit ({age_mins:.0f} min) — closing")
            _close_signal(sb, sig["id"], "time_limit",
                          current_price=price,
                          entry_price=float(sig.get("entry_price") or 0),
                          direction=direction)
            _log_event(sb, sig["id"], "time_limit",
                       price=price,
                       note=f"30-min scalp window expired — position exited at ${price:.2f}" if price else
                            "30-min scalp window closed — position exited")
            try:
                _push_scalp_expired(ticker, direction)
            except Exception:
                pass
            continue

        # ── 2a. EOD early warning (3:00–3:30 PM) — alert if in profit ────────
        if eod_warning and strategy in INTRADAY_STRATEGIES:
            try:
                price = _current_price(ticker)
                entry = float(sig.get("entry_price") or 0)
                if price and entry:
                    is_long = direction == "LONG"
                    pnl_pct = ((price - entry) / entry * 100) if is_long else ((entry - price) / entry * 100)
                    if pnl_pct > 0.5:   # only warn if meaningfully in profit (>0.5%)
                        logger.info(f"[monitor] {ticker} EOD warning — in profit {pnl_pct:.1f}%")
                        _push_eod_warning(ticker, direction, pnl_pct)
            except Exception:
                pass
            # Do NOT continue — still run normal checks below

        # ── 2b. Market-close force-close (3:30 PM+) — exit all intraday ──────
        if near_close and strategy in INTRADAY_STRATEGIES:
            price = _current_price(ticker)
            entry = float(sig.get("entry_price") or 0)
            is_long = direction == "LONG"
            pnl_pct = None
            if price and entry:
                raw = ((price - entry) / entry * 100)
                pnl_pct = raw if is_long else -raw

            logger.info(
                f"[monitor] {ticker} [{strategy}] force-close at market end "
                f"price={price} pnl={pnl_pct:.1f}%" if pnl_pct is not None else
                f"[monitor] {ticker} [{strategy}] force-close at market end"
            )
            _close_signal(sb, sig["id"], "market_close",
                          current_price=price,
                          entry_price=entry,
                          direction=direction)
            pnl_str = f"+{pnl_pct:.1f}%" if pnl_pct and pnl_pct > 0 else f"{pnl_pct:.1f}%" if pnl_pct else ""
            _log_event(sb, sig["id"], "market_close",
                       price=price,
                       note=f"Market closing — {direction} exited at ${price:.2f} {pnl_str}".strip()
                            if price else f"Market closing — {direction} position force-closed")
            try:
                _push_market_close(ticker, direction, strategy, pnl_pct)
            except Exception:
                pass
            continue

        # Only run analysis checks during market hours
        if not market_open:
            continue

        # ── Get price + levels once for all checks below ──────────────────────
        try:
            entry  = float(sig["entry_price"])
            t1     = float(sig["target_one"])
            t2     = float(sig["target_two"])
            sl     = float(sig["stop_loss"])
            price  = _current_price(ticker)
        except Exception as e:
            logger.debug(f"[monitor] Level parse error for {ticker}: {e}")
            continue

        if not price:
            continue

        is_long  = direction == "LONG"
        pnl_pct  = ((price - entry) / entry * 100) if is_long else ((entry - price) / entry * 100)

        # ── 3. Status change tracking + event logging ─────────────────────────
        try:
            new_status = _derive_status(price, entry, t1, sl, direction)
            old_status = _STATUS_CACHE.get(sig["id"])

            if new_status != old_status:
                _STATUS_CACHE[sig["id"]] = new_status
                _log_status_event(sb, sig["id"], new_status, price,
                                  extra=f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%")
                # Push notification for important transitions only
                push_statuses = {"near_stop", "strong_profit", "at_target"}
                if new_status in push_statuses:
                    try:
                        _push_status_change(ticker, new_status, pnl_pct)
                    except Exception:
                        pass
                logger.info(
                    f"[monitor] {ticker} status: {old_status} → {new_status} "
                    f"price={price:.2f} pnl={pnl_pct:+.2f}%"
                )
        except Exception as e:
            logger.debug(f"[monitor] Status tracking error for {ticker}: {e}")

        # ── 4. T1 hit → move SL to breakeven ─────────────────────────────────
        try:
            if sl != entry:   # not already at breakeven
                t1_hit = (is_long and price >= t1) or (not is_long and price <= t1)
                t2_hit = (is_long and price >= t2) or (not is_long and price <= t2)

                if t1_hit and not t2_hit:
                    pct = abs(price - entry) / entry * 100
                    logger.info(f"[monitor] {ticker} T1 hit @ {price:.2f} — moving SL to breakeven")
                    _update_sl(sb, sig["id"], entry)
                    _log_event(sb, sig["id"], "t1_hit", price=price,
                               note=f"🎯 T1 hit @ ${price:.2f} (+{pct:.1f}%) — stop moved to breakeven ${entry:.2f}")
                    try:
                        _push_t1_breakeven(ticker, direction, pct)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"[monitor] T1 check error for {ticker}: {e}")

        # ── 5. Intelligent early booking ──────────────────────────────────────
        # Book profit early when quant conditions indicate momentum is failing.
        # Only applies to in-profit signals that are building/strong profit.
        try:
            current_status = _STATUS_CACHE.get(sig["id"], "")
            should_assess  = current_status in ("building_profit", "strong_profit") and pnl_pct > 0.5

            if should_assess:
                book_now, reason = _momentum_check(ticker, direction)

                # Book early if signal has been stalling for >3h with no progress
                # (was 2h — raised to 3h to give trades more room to breathe)
                if not book_now and age_mins > 180 and current_status == "building_profit":
                    book_now = True
                    reason   = f"Signal stalling after {age_mins:.0f} min — booking profit to protect gains"

                # REMOVED: 2PM blanket auto-book.
                # Closing at +0.2% because the clock hit 2 PM created hundreds of
                # microscopic "wins" that masked real loss performance. Trades
                # that need to run to T1 should be allowed to run. The EOD
                # force-close at 3:30 PM provides the real time-based exit.

                if book_now:
                    # Only auto-close if profit is meaningful (≥1.0%) — tiny wins
                    # at +0.2% mask real performance and don't cover losses.
                    # Below 1.0%, send a push notification but don't auto-close.
                    if pnl_pct >= 1.0:
                        logger.info(f"[monitor] {ticker} EARLY BOOK — {reason} pnl={pnl_pct:.1f}%")
                        _close_signal(sb, sig["id"], "target_hit",
                                      current_price=price, entry_price=entry, direction=direction)
                        _log_event(sb, sig["id"], "closed_win", price=price,
                                   note=f"💡 Profit booked @ ${price:.2f} (+{pnl_pct:.1f}%) — {reason}")
                        _STATUS_CACHE.pop(sig["id"], None)
                        try:
                            _push_early_book(ticker, direction, pnl_pct, reason)
                        except Exception:
                            pass
                        continue   # signal is now closed
                    else:
                        logger.debug(f"[monitor] {ticker} early book skipped — pnl {pnl_pct:.1f}% < 1.0% minimum")
        except Exception as e:
            logger.debug(f"[monitor] Early booking check error for {ticker}: {e}")

        # ── 6. Structure reversal detection ───────────────────────────────────
        try:
            if _detect_structure_reversal(ticker, direction):
                logger.info(f"[monitor] {ticker} structure reversed — closing {direction} early")
                _close_signal(sb, sig["id"], "structure_reversal",
                              current_price=price, entry_price=entry, direction=direction)
                opposite = "bearish" if direction == "LONG" else "bullish"
                _log_event(sb, sig["id"], "reversal", price=price,
                           note=f"⚠️ {opposite.capitalize()} CHoCH detected — {direction} closed early @ ${price:.2f} ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%)")
                _STATUS_CACHE.pop(sig["id"], None)
                try:
                    _push_reversal(ticker, direction)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"[monitor] Reversal check error for {ticker}: {e}")


# ---------------------------------------------------------------------------
# Options signal monitor
# ---------------------------------------------------------------------------

def _monitor_options(sb: Client) -> None:
    try:
        rows = (
            sb.table("option_signals")
            .select("*")
            .eq("status", "active")
            .execute()
            .data
        ) or []
    except Exception as e:
        logger.error(f"[monitor] Failed to fetch option signals: {e}")
        return

    if not rows:
        return

    logger.info(f"[monitor] Checking {len(rows)} active option signal(s)")
    near_close  = _is_near_market_close()
    market_open = _is_market_hours()
    now_utc     = datetime.now(timezone.utc)

    for sig in rows:
        ticker   = sig["ticker"]
        strategy = sig.get("strategy_type") or "day_trade"
        direction = sig.get("direction", "LONG")
        created  = datetime.fromisoformat(sig["created_at"].replace("Z", "+00:00"))
        age_mins = (now_utc - created).total_seconds() / 60

        # ── 1. Market-close enforcer ──────────────────────────────────────────
        # Options day trades MUST be closed before market close — they lose
        # all time value rapidly in the last 10 minutes and spreads widen.
        if near_close and strategy in INTRADAY_STRATEGIES:
            logger.info(f"[monitor] {ticker} [option/{strategy}] market closing — force close")
            _close_signal(sb, sig["id"], "market_close", close_type="option")
            _log_event(sb, sig["id"], "market_close",
                       note=f"Market closing — option position closed at 3:50 PM ET")
            try:
                push._send_raw(
                    title=f"⏰ Close Option Now — {ticker}",
                    body=f"Market closing in 10 min. Exit your {strategy.replace('_',' ')} option position.",
                    data={"type": "market_close", "ticker": ticker, "signal_type": "option"},
                )
            except Exception:
                pass
            continue

        # ── 2. Swing option — expired DTE ────────────────────────────────────
        # If the option's expiry date has passed → force close
        try:
            expiry_str = sig.get("expiry_date")
            if expiry_str:
                expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if now_utc >= expiry_dt:
                    logger.info(f"[monitor] {ticker} option expired (DTE=0) — closing")
                    _close_signal(sb, sig["id"], "option_expired", close_type="option")
                    _log_event(sb, sig["id"], "option_expired",
                               note=f"{sig.get('contract_type','?')} option reached expiry date")
                    try:
                        push._send_raw(
                            title=f"⚠ Option Expired — {ticker}",
                            body=f"Your {sig.get('contract_type','?')} option has reached expiry.",
                            data={"type": "option_expired", "ticker": ticker},
                        )
                    except Exception:
                        pass
                    continue
        except Exception:
            pass

        if not market_open:
            continue

        # ── 3. Underlying reversal → close option ────────────────────────────
        # If the underlying stock reverses structure, the option loses value fast.
        try:
            if _detect_structure_reversal(ticker, direction):
                logger.info(f"[monitor] {ticker} underlying reversed — closing option {direction} early")
                _close_signal(sb, sig["id"], "structure_reversal", close_type="option")
                try:
                    push._send_raw(
                        title=f"⚠ Underlying Reversed — {ticker}",
                        body=f"Stock structure changed against your {direction} option. Consider closing.",
                        data={"type": "reversal", "ticker": ticker, "signal_type": "option"},
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"[monitor] Option reversal check error for {ticker}: {e}")

        # ── 4. Premium level check ────────────────────────────────────────────
        # Estimate current premium via delta × underlying move
        # Close if estimated premium hit target or stop
        try:
            price      = _current_price(ticker)
            delta      = float(sig.get("delta") or 0)
            und_price  = float(sig.get("underlying_price") or 0)
            entry_prem = float(sig.get("entry_premium") or 0)
            target_prem = float(sig.get("target_premium") or 0)
            stop_prem   = float(sig.get("stop_premium") or 0)

            if price and und_price and delta and entry_prem:
                est_prem = entry_prem + delta * (price - und_price)
                result   = None

                if est_prem >= target_prem:
                    result = "win"
                elif est_prem <= stop_prem:
                    result = "loss"

                if result:
                    pct = ((est_prem - entry_prem) / entry_prem * 100) if entry_prem else 0
                    sb.table("option_signals").update({
                        "status":        "closed",
                        "closed_reason": "target_hit" if result == "win" else "stop_hit",
                        "result":        result,
                        "closed_at":     now_utc.isoformat(),
                    }).eq("id", sig["id"]).execute()
                    event_type = "closed_win" if result == "win" else "closed_loss"
                    event_note = (
                        f"Target hit — premium ${entry_prem:.2f}→${est_prem:.2f} (+{pct:.1f}%)"
                        if result == "win"
                        else f"Stop hit — premium ${entry_prem:.2f}→${est_prem:.2f} ({pct:.1f}%)"
                    )
                    _log_event(sb, sig["id"], event_type, price=price, note=event_note)
                    logger.info(f"[monitor] Option {ticker} {result} — premium {entry_prem:.2f}→{est_prem:.2f} ({pct:+.1f}%)")
                    try:
                        _push_closed(ticker, direction, result, pct)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"[monitor] Option premium check error for {ticker}: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Full monitoring pass — stocks + options.
    Called every 15 minutes from runner.py maintenance job.
    """
    logger.info(
        f"[monitor] Pass started — "
        f"ET={_now_et().strftime('%H:%M')} "
        f"near_close={_is_near_market_close()} "
        f"market_open={_is_market_hours()}"
    )

    sb = _supabase()
    _monitor_stocks(sb)
    _monitor_options(sb)

    logger.info("[monitor] Pass complete")
