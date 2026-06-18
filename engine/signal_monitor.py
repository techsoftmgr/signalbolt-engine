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
from engine import exit_intelligence
from engine import trend_ride

logger = logging.getLogger("signalbolt.monitor")

# ── In-memory status cache ────────────────────────────────────────────────────
# Tracks last known status per signal_id so we only log events on transitions.
# Lost on engine restart — that's fine, the signal_events table is the truth.
# Structure: { signal_id: "near_stop" | "below_entry" | "in_profit" |
#                         "building_profit" | "strong_profit" | "at_target" }
_STATUS_CACHE: dict[str, str] = {}
# Track which signals have already received an EOD warning this session
_EOD_WARNED: set[str] = set()
# Swing trailing stop — tracks the best (peak) price seen after T1 hit per signal.
# Updated every monitor pass while the swing signal is between T1 and T2.
# Evicted when the signal closes via any path.
_SWING_PEAK: dict[str, float] = {}   # signal_id → peak price after T1

ET = ZoneInfo("America/New_York")

# ── Strategy close rules ──────────────────────────────────────────────────────
# Strategies that must be closed by market close (not held overnight)
INTRADAY_STRATEGIES = {"scalping", "day_trade", "options_flow"}

# Multi-day / overnight (SWING-class) strategies — EXCLUDED from the intraday
# early-booking block (step 5). They ride to T1/T2 + trailing + structure-reversal
# instead; subjecting them to intraday RSI/MACD + a CALENDAR-minute "stalling >180"
# rule books them prematurely (an overnight hold trivially exceeds 180 min). Before
# this, only the literal "swing_trade" was excluded, so breakdown/breakout/peak/
# turnaround/*_forming/position slipped through and got booked the morning after —
# e.g. AMZN breakdown booked +1.2% instead of riding toward its target.
_SWING_LIKE_STRATEGIES = {
    "swing_trade", "breakdown", "breakout", "turnaround", "peak",
    "breakdown_forming", "distrib_forming", "peak_forming", "turn_forming",
    "accum_forming", "position_trade",
}

# Scalping max hold in minutes (regardless of market hours)
SCALP_MAX_HOLD_MINS = 30

# ── Dynamic trailing stop (after T1) ──────────────────────────────────────────
# After T1 hits, trail the visible stop up — staying a fixed % BELOW the
# running peak so it tracks price closely and locks in profit as the move
# extends. Ratchets up only, never below breakeven. Activates once price is
# TRAIL_MIN_MOVE_PCT beyond T1.
#
# % below peak is per-strategy (wider for slower timeframes that breathe more):
TRAIL_PEAK_PCT = {
    "scalping":     0.004,   # 0.4% below peak — tight, fast exits
    "day_trade":    0.010,   # 1.0% below peak
    "options_flow": 0.010,
    "dark_pool":    0.010,
    "swing_trade":  0.020,   # 2.0% below peak — room for multi-day breathing
    "vwap_reclaim": 0.010,
    "breakdown":    0.025,   # daily swing short — extra room (far T1, ride bounces)
    "breakout":     0.025,   # daily swing long
    "turnaround":   0.025,   # daily swing long off a cycle bottom
    "peak":         0.025,   # daily swing short off a cycle top
}
TRAIL_DEFAULT_PCT  = 0.012
TRAIL_MIN_MOVE_PCT = 0.005  # peak must be ≥0.5% beyond T1 before trailing starts

# Pre-T1 breakeven protection: if a profitable position shows MODERATE reversal
# pressure (this score ≤ pressure < exit threshold of 55) we don't book early,
# but we move the stop to breakeven to protect the gain while it rides. Below
# this we leave the stop alone so a clean trend isn't whipsawed.
_BE_PROTECT_SCORE = 40

# Pre-T1 peak trailing: start the peak-based trailing stop once peak profit has
# covered this fraction of the entry->T1 distance — even before T1. Protects
# swings that run most of the way then stall (DVN ran 84% to T1, then chopped
# for 2 days and expired ~flat). Floored at breakeven, ratchets up only, uses
# the loose per-strategy TRAIL_PEAK_PCT so swings aren't whipsawed.
_TRAIL_ACTIVATE_FRAC = 0.6
# Absolute-profit early lock: once a position is up at least this %, ratchet the
# stop (floored at breakeven) EVEN if it's nowhere near T1 — so a far-target
# swing (breakdown short: T1 ≈ -1.5 ATR ≈ -10%+) doesn't round-trip a +2-3% gain
# while waiting for the 60%-to-T1 trail. Tightens the stop only; never closes.
_BE_PROFIT_PCT = 2.0

# Breakdown/breakout target a FAR T1 (≈ ±5%+). At a modest +2% gain a tight
# peak-trail gets knocked out by a NORMAL bounce before the move plays out
# (CME breakdown: +2.3% peak, ~2.3% bounce → stopped at +0.4%). So for these,
# between _BE_PROFIT_PCT and this %, move the stop only to BREAKEVEN (room to
# ride, no loss); switch to the tight peak-trail once genuinely profitable.
# Other strategies (no entry) keep the original "tight-trail at +2%" behaviour.
_LOCK_TIGHT_PCT = {"breakdown": 3.0, "breakout": 3.0, "turnaround": 3.0, "peak": 3.0}
# Near-expiry profit backstop: within this many hours of max-hold, book a still-
# green position at market rather than let it ride to a flat/near-flat expiry.
_NEAR_EXPIRY_HRS        = 3.0
_NEAR_EXPIRY_MIN_PROFIT = 0.8   # only backstop if at least this % in profit

# Time-stop (intraday): SMC winners resolved fast (~31m median) while losers
# dragged ~90m+ to a full stop (May 27-28 analysis). If an intraday signal hasn't
# made meaningful progress (this % in profit) by _TIME_STOP_MINS, cut it now —
# turns slow -1.5% bleeders into small scratches. Intraday strategies only.
_TIME_STOP_MINS         = 45
_TIME_STOP_MIN_PROGRESS = 0.2
_TIME_STOP_STRATEGIES   = {"day_trade", "scalping"}

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


# ── MFE / MAE excursion capture ───────────────────────────────────────────────

def _in_quote_window() -> bool:
    """RTH + extended hours (4:00 AM – 8:00 PM ET, Mon–Fri) — the window where
    live prints exist for excursion capture. Skips overnight/weekends (no fresh
    trades → nothing to record)."""
    now = _now_et()
    if now.weekday() >= 5:
        return False
    return 4 <= now.hour < 20


def _mins_since_entry(created_at) -> float | None:
    """Minutes from a signal's entry timestamp to now (UTC). None if unparseable."""
    if not created_at:
        return None
    try:
        from datetime import datetime, timezone
        c = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if c.tzinfo is None:
            c = c.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - c).total_seconds() / 60.0, 1)
    except Exception:
        return None


def _capture_excursion(sb, sig) -> None:
    """Record running MFE (peak favorable) / MAE (peak adverse) unrealized % on a
    signal — INCLUDING the big after-hours / pre-market excursions — so profit
    GIVE-BACK (peak − realized) is measurable per signal. MEASUREMENT ONLY: never
    closes or moves a stop. Stored in score_breakdown.{mfe_pct,mae_pct} plus the
    TIMING of each extreme ({t_mfe_min,t_mae_min,mae_before_mfe}) — the raw
    material for profit-lock + stop tuning (JSONB, no migration); the row is
    written only when a NEW extreme is set."""
    try:
        entry = float(sig.get("entry_price") or 0)
    except (TypeError, ValueError):
        return
    if entry <= 0:
        return
    ticker = sig.get("ticker")
    price  = _current_price(ticker)
    if not price or price <= 0:
        return
    is_long = (sig.get("direction") or "").upper() == "LONG"
    pnl = ((price - entry) / entry * 100.0) if is_long else ((entry - price) / entry * 100.0)
    # Phantom-print guard for MEASUREMENT: a single bad SIP tick shouldn't inflate
    # the recorded peak. A real earnings gap is well under this; >60% in one read
    # is almost always a bad print.
    if abs(pnl) > 60.0:
        return
    sbd     = sig.get("score_breakdown") or {}
    cur_mfe = sbd.get("mfe_pct")
    cur_mae = sbd.get("mae_pct")
    new_mfe = pnl if cur_mfe is None else max(float(cur_mfe), pnl)
    new_mae = pnl if cur_mae is None else min(float(cur_mae), pnl)
    mfe_changed = new_mfe != cur_mfe
    mae_changed = new_mae != cur_mae
    if not mfe_changed and not mae_changed:
        return  # no new extreme — skip the write
    merged = dict(sbd)
    merged["mfe_pct"] = round(new_mfe, 2)
    merged["mae_pct"] = round(new_mae, 2)
    # TIMING — minutes from entry to each extreme + which came FIRST. Answers the
    # two exit-tuning questions: profit-lock (when does favorable excursion peak?
    # → t_mfe_min) and min-stop (do winners take heat BEFORE they work? →
    # mae_before_mfe). Stamped on the read that sets each new extreme.
    elapsed = _mins_since_entry(sig.get("created_at"))
    if elapsed is not None:
        if mfe_changed:
            merged["t_mfe_min"] = elapsed
        if mae_changed:
            merged["t_mae_min"] = elapsed
        t_mfe, t_mae = merged.get("t_mfe_min"), merged.get("t_mae_min")
        if t_mfe is not None and t_mae is not None:
            merged["mae_before_mfe"] = bool(t_mae < t_mfe)
    try:
        sb.table("signals").update({"score_breakdown": merged}).eq("id", sig["id"]).execute()
        sig["score_breakdown"] = merged   # keep local copy fresh for downstream checks
    except Exception as e:
        logger.debug(f"[monitor] MFE/MAE update failed for {ticker}: {e}")


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
#
# Status labels are DIRECTION-AWARE.
# "below_entry" means "price moved the wrong way from entry" for both directions:
#   LONG:  price dropped below entry  → "Below Entry — Waiting for recovery"
#   SHORT: price rose above entry     → "Above Entry — Waiting for reversal"
# "in_profit" also uses direction-specific wording so the insight line reads
# naturally for SHORT traders (dropping = profit, not a negative move).

_STATUS_LABELS_LONG = {
    "near_stop":       ("⚠️", "Near Stop — Watch closely"),
    "below_entry":     ("↩",  "Below Entry — Waiting for recovery"),
    "in_profit":       ("💹", "In Profit — Hold, target not reached"),
    "building_profit": ("📈", "Building Profit — Rising toward target"),
    "strong_profit":   ("🔥", "Strong Profit — Consider booking partial"),
    "at_target":       ("🎯", "At Target — Book profit"),
}

_STATUS_LABELS_SHORT = {
    "near_stop":       ("⚠️", "Near Stop — Watch closely"),
    "below_entry":     ("↩",  "Above Entry — Waiting for reversal"),   # price above entry = against SHORT
    "in_profit":       ("💹", "In Profit — Hold, target not reached"),
    "building_profit": ("📈", "Building Profit — Dropping toward target"),
    "strong_profit":   ("🔥", "Strong Profit — Consider booking partial"),
    "at_target":       ("🎯", "At Target — Book profit"),
}

# Legacy alias used by _push_status_change (direction-neutral fallback)
_STATUS_LABELS = _STATUS_LABELS_LONG


def _get_labels(direction: str) -> dict:
    return _STATUS_LABELS_SHORT if direction == "SHORT" else _STATUS_LABELS_LONG


def _log_status_event(
    sb: Client, sig_id: str, status: str,
    price: float | None, direction: str = "LONG",
    extra: str = "", stop_loss: float | None = None,
) -> None:
    """Log a status-change event to signal_events timeline."""
    emoji, base_label = _get_labels(direction).get(status, ("•", status))
    note = f"{emoji} {base_label}"
    if extra:
        note += f" — {extra}"
    if price:
        note += f" (${price:.2f})"
    # For near_stop events: show the stop level and whether price has crossed it
    if status == "near_stop" and price is not None and stop_loss is not None:
        is_long = direction == "LONG"
        dist = price - stop_loss if is_long else stop_loss - price
        if dist < 0:
            note += f" · ⚠️ Stop ${stop_loss:.2f} already crossed"
        else:
            note += f" · Stop ${stop_loss:.2f} (${dist:.2f} away)"
    _log_event(sb, sig_id, status, price=price, note=note)


def _close_signal(
    sb: Client,
    sig_id: str,
    reason: str,
    close_type: str = "stock",
    current_price: float | None = None,
    entry_price: float | None = None,
    direction: str = "LONG",
    ticker: str | None = None,
) -> None:
    """
    Write closed status to Supabase.
    When current_price + entry_price are provided (e.g. market_close, time_limit)
    the actual P&L is recorded so history shows win/loss — not just 'expired'.

    When `ticker` is given, the recorded price is run through
    alpaca_client.sane_close_price() so a single bad SIP print can't mis-record
    P&L on a non-level close (EOD / time-stop / near-expiry / trend exit). Level
    closes (stop/target) already cap at the level upstream.
    """
    table = "option_signals" if close_type == "option" else "signals"

    # Bad-print guard for stock closes that book a live market price.
    if ticker and current_price is not None and close_type != "option":
        current_price = _alpaca.sane_close_price(ticker, current_price)

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


def _update_sl(sb: Client, sig_id: str, new_sl: float, sig: dict | None = None) -> None:
    """Move stop loss to new level (e.g. breakeven after T1 hit).

    If `sig` (the pre-update signal row) is passed, also push a ONE-TIME alert
    when this update is the moment the stop first crosses to breakeven-or-better
    (old stop on the risk side of entry, new stop at/beyond entry). Because the
    stop only ratchets in one direction, that crossing happens on exactly one
    update — so the notification fires once, with no dedup state needed.
    """
    try:
        sb.table("signals").update({
            "stop_loss": round(new_sl, 4),
        }).eq("id", sig_id).execute()
    except Exception as e:
        logger.error(f"[monitor] SL update failed for {sig_id}: {e}")
        return

    if not sig:
        return
    try:
        entry  = float(sig.get("entry_price") or 0)
        old_sl = float(sig.get("stop_loss") or 0)
        if entry <= 0:
            return
        is_long = sig.get("direction") == "LONG"
        eps = max(0.01, entry * 0.0005)
        crossed = ((is_long and old_sl < entry - eps and new_sl >= entry - eps) or
                   (not is_long and old_sl > entry + eps and new_sl <= entry + eps))
        if crossed:
            locked = abs(new_sl - entry) / entry * 100
            from engine import push
            push.send_stop_protected_alert(
                sig.get("ticker", ""), sig.get("direction", "LONG"),
                round(new_sl, 2), round(locked, 1), signal_id=str(sig_id),
            )
            logger.info(f"[monitor] {sig.get('ticker')} stop crossed to B/E "
                        f"({old_sl:.2f} -> {new_sl:.2f}, entry {entry:.2f}) — pushed")
    except Exception as e:
        logger.debug(f"[monitor] stop-protected push check failed for {sig_id}: {e}")


def _set_trend_ride_flag(sb: Client, sig: dict, on: bool) -> None:
    """Persist the trend_ride state inside score_breakdown so (a) the early-exit paths can
    tell a riding swing apart and (b) the feature's effect is measurable.
      • trend_ride       — LIVE "currently riding" flag (set/cleared as the ride starts/ends)
      • trend_ride_ever  — SET-ONCE durable marker (never cleared) so a CLOSED row still records
                           that it rode even after the live flag was cleared on the trend_break
                           exit — this is what the trend_ride scorecard segments on.
    Idempotent — only writes when something actually changes. Mutates `sig` in place so the
    rest of this pass sees the new state."""
    bd = dict(sig.get("score_breakdown") or {})
    changed = False
    if bool(bd.get("trend_ride")) != bool(on):
        bd["trend_ride"] = bool(on)
        changed = True
    if on and not bd.get("trend_ride_ever"):
        bd["trend_ride_ever"] = True
        changed = True
    if not changed:
        return
    sig["score_breakdown"] = bd
    try:
        sb.table("signals").update({"score_breakdown": bd}).eq("id", sig["id"]).execute()
    except Exception as e:
        logger.debug(f"[monitor] trend_ride flag update failed for {sig.get('id')}: {e}")


# ── Corporate-action (split) guard ──────────────────────────────────────────
# Alpaca bars are split-ADJUSTED (alpaca_client get_bars/get_latest), but a stored
# signal's price levels are the NOMINAL values captured at entry. After a split the
# live price jumps scale (e.g. KLAC 10:1, 2026-06-12: entry ~$2,000 → price ~$200)
# and every downstream check would book a phantom -90% stop. We detect a CONFIRMED
# split and rescale the levels instead — never recording a phantom close.
_SPLIT_RATIOS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30)


def detect_split_factor(old_px, new_px, tol: float = 0.06):
    """If old_px and new_px look like the same security on two different split
    scales, return the split factor F such that new ≈ old / F — forward split F>1
    (price fell ~F×), reverse split 0<F<1 (price rose ~1/F×). Else None. Pure; F is
    drawn from a curated set of clean ratios matched within `tol`. NOTE: a price that
    merely halved/tripled also matches here — callers must CONFIRM against split-
    adjusted history (see _confirm_split_factor) so a real crash never qualifies."""
    try:
        old_px = float(old_px); new_px = float(new_px)
    except (TypeError, ValueError):
        return None
    if old_px <= 0 or new_px <= 0:
        return None
    ratio = old_px / new_px                       # ≈ F for a forward split
    for n in _SPLIT_RATIOS:
        if abs(ratio - n) <= n * tol:             # forward split n:1
            return float(n)
        if abs(ratio - 1.0 / n) <= (1.0 / n) * tol:   # reverse split 1:n
            return 1.0 / n
    return None


def _confirm_split_factor(sig: dict, current_price: float | None):
    """Return the split factor ONLY if confirmed by split-adjusted history: the
    stored (nominal) entry must be a clean split-multiple of the split-ADJUSTED close
    on the signal's own entry date. A real price crash leaves that comparison ≈1, so
    this fires on an actual split/reverse-split and never on a collapse. The current-
    price gap is just a cheap pre-filter to avoid fetching bars every cycle."""
    try:
        entry = float(sig.get("entry_price") or 0)
    except (TypeError, ValueError):
        return None
    if entry <= 0 or not current_price:
        return None
    if detect_split_factor(entry, current_price) is None:   # cheap pre-filter
        return None
    try:
        created = datetime.fromisoformat(sig["created_at"].replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - created).days
        df = _alpaca.get_bars(sig["ticker"], "1Day", days=min(age_days + 6, 400))
        if df is None or len(df) == 0:
            return None
        d0 = created.date()
        adj_entry_close = None
        for ts, close in zip(df.index, df["close"]):
            if ts.date() <= d0:
                adj_entry_close = float(close)
            else:
                break
        if adj_entry_close is None:
            return None
        return detect_split_factor(entry, adj_entry_close)
    except Exception as e:
        logger.debug(f"[monitor] split-confirm failed for {sig.get('ticker')}: {e}")
        return None


def _apply_split_adjustment(sb: Client, sig: dict, factor: float) -> None:
    """Rescale an open signal's price levels by a confirmed split factor so it keeps
    tracking correctly (forward F>1 divides; reverse F<1 multiplies)."""
    def _adj(v):
        try:
            return round(float(v) / factor, 4) if v is not None else None
        except (TypeError, ValueError):
            return None
    payload = {}
    for col in ("entry_price", "stop_loss", "target_one", "target_two"):
        nv = _adj(sig.get(col))
        if nv is not None:
            payload[col] = nv
    if not payload:
        return
    try:
        sb.table("signals").update(payload).eq("id", sig["id"]).execute()
        logger.warning(
            f"[monitor] {sig.get('ticker')} SPLIT detected (factor {factor:g}) — rescaled "
            f"levels (entry {sig.get('entry_price')} -> {payload.get('entry_price')}); "
            f"no phantom close"
        )
    except Exception as e:
        logger.error(f"[monitor] split adjust failed for {sig.get('id')}: {e}")


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

def _push_market_close(
    ticker: str, direction: str, strategy: str,
    pnl_pct: float | None = None, signal_id: str | None = None,
    created_at: str | None = None,
) -> None:
    if pnl_pct is not None and pnl_pct > 0:
        title = f"✅ {ticker} closed +{pnl_pct:.1f}% — Market Close"
        body  = f"Booked profit on {direction} {strategy.replace('_',' ')} before 4 PM ET"
    elif pnl_pct is not None and pnl_pct <= 0:
        title = f"⏰ {ticker} closed {pnl_pct:.1f}% — Market Close"
        body  = f"Position exited to avoid overnight risk. {direction} {strategy.replace('_',' ')}"
    else:
        title = f"⏰ Market Closing — {ticker}"
        body  = f"Close your {direction} {strategy.replace('_',' ')} position before 4 PM ET"
    data: dict = {"type": "market_close", "ticker": ticker}
    if signal_id:
        data["signal_id"] = signal_id
    if created_at:
        data["created_at"] = created_at
    push._send_raw(title=title, body=body, data=data)


def _push_eod_warning(ticker: str, direction: str, pnl_pct: float, signal_id: str | None = None) -> None:
    now_et    = datetime.now(ZoneInfo("America/New_York"))
    mins_left = max(0, (16 * 60) - (now_et.hour * 60 + now_et.minute))
    data: dict = {"type": "eod_warning", "ticker": ticker}
    if signal_id:
        data["signal_id"] = signal_id
    push._send_raw(
        title=f"📊 {ticker} +{pnl_pct:.1f}% — {mins_left} min to close",
        body=f"Your {direction} signal is in profit. Consider booking before market close.",
        data=data,
    )


def _push_early_book(
    ticker: str, direction: str, pnl_pct: float, reason: str, signal_id: str | None = None,
) -> None:
    data: dict = {"type": "book_profit", "ticker": ticker}
    if signal_id:
        data["signal_id"] = signal_id
    push._send_raw(
        title=f"💡 Book Profit Now — {ticker} +{pnl_pct:.1f}%",
        body=reason,
        data=data,
    )


def _push_status_change(
    ticker: str, status: str, pnl_pct: float | None,
    direction: str = "LONG", signal_id: str | None = None,
) -> None:
    emoji, label = _get_labels(direction).get(status, ("•", status))
    pnl_str = f" ({'+' if pnl_pct and pnl_pct > 0 else ''}{pnl_pct:.1f}%)" if pnl_pct is not None else ""
    data: dict = {"type": "status_change", "ticker": ticker, "status": status}
    if signal_id:
        data["signal_id"] = signal_id
    push._send_raw(
        title=f"{emoji} {ticker}{pnl_str} — {label.split(' — ')[0]}",
        body=label.split(" — ", 1)[-1] if " — " in label else label,
        data=data,
    )


def _push_reversal(ticker: str, direction: str, signal_id: str | None = None) -> None:
    opposite = "BEARISH" if direction == "LONG" else "BULLISH"
    data: dict = {"type": "reversal", "ticker": ticker, "direction": direction}
    if signal_id:
        data["signal_id"] = signal_id
    push._send_raw(
        title=f"⚠ Structure Reversed — {ticker}",
        body=f"{opposite} CHoCH detected. Consider closing your {direction} position.",
        data=data,
    )


def _push_t1_breakeven(ticker: str, direction: str, pct: float, signal_id: str | None = None) -> None:
    data: dict = {"type": "t1_breakeven", "ticker": ticker}
    if signal_id:
        data["signal_id"] = signal_id
    push._send_raw(
        title=f"🎯 T1 Hit — {ticker} +{pct:.1f}%",
        body=f"Stop moved to breakeven. Riding to T2. {direction} still open.",
        data=data,
    )


def _push_scalp_expired(ticker: str, direction: str, signal_id: str | None = None) -> None:
    data: dict = {"type": "scalp_expired", "ticker": ticker}
    if signal_id:
        data["signal_id"] = signal_id
    push._send_raw(
        title=f"⏱ Scalp Time Limit — {ticker}",
        body=f"30-min scalp window closed. Exit your {direction} position now.",
        data=data,
    )


def _push_closed(ticker: str, direction: str, result: str, pct: float,
                 created_at: str | None = None, is_option: bool = False,
                 contract: str | None = None, signal_id: str | None = None) -> None:
    # Option closes MUST read as option closes. An option's premium-based P&L
    # (e.g. -28.3% on a ~-1% underlying move, via delta) otherwise looks like it
    # belongs to the STOCK signal on the same ticker — which may still be open or
    # only moved -1.4% (the IWM confusion). Label "Option", name the contract, and
    # tag signal_type so the tap routes to the Options tab.
    kind = "Option " if is_option else ""
    subj = (f"{ticker} {contract}" if (is_option and contract) else f"{direction} signal")
    data: dict = {"type": "signal_closed", "result": result, "ticker": ticker,
                  "created_at": created_at}
    if is_option:
        data["signal_type"] = "option"
    if signal_id:
        data["signal_id"] = signal_id
    if result == "win":
        push._send_raw(
            title=f"✅ {kind}Target Hit — {ticker} +{pct:.1f}%",
            body=f"{subj} closed with a win.",
            data=data,
        )
    elif result == "loss":
        push._send_raw(
            title=f"🔴 {kind}Stop Hit — {ticker} {pct:.1f}%",
            body=f"{subj} stopped out.",
            data=data,
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
    near_close     = _is_near_market_close()
    eod_warning    = _is_eod_warning()
    market_open    = _is_market_hours()
    capture_window = _in_quote_window()   # RTH + extended hours
    now_utc        = datetime.now(timezone.utc)

    for sig in rows:
        # ── MFE/MAE capture (ALL active signals incl. manual/momentum; RTH +
        #    extended hours). Records running peak-favorable / peak-adverse
        #    unrealized % so give-back = peak − realized is measurable per signal,
        #    INCLUDING the big after-hours / pre-market runs. Measurement only —
        #    runs before every skip so coverage is complete, never trades. ──
        if capture_window:
            try:
                _capture_excursion(sb, sig)
            except Exception:
                pass

        # MANUAL override: the admin owns this signal. The engine must not trail,
        # move the stop, EOD-close, time-stop, or reverse-exit it — nothing — until
        # it's flipped back to engine management. (Manual create + agentic control.)
        if (sig.get("management_mode") or "engine") == "manual":
            continue
        # TREND_MOMENTUM signals are fully owned by engine.momentum_monitor
        # (chandelier trail, daily-close trend-break exit, no fixed targets /
        # breakeven / EOD). Skip them here so the two managers don't conflict.
        if ((sig.get("score_breakdown") or {}).get("detector_source")) == "TREND_MOMENTUM":
            continue

        ticker    = sig["ticker"]
        strategy  = sig.get("strategy_type") or "day_trade"
        direction = sig["direction"]
        created   = datetime.fromisoformat(sig["created_at"].replace("Z", "+00:00"))
        age_mins  = (now_utc - created).total_seconds() / 60

        sig_id_str = str(sig["id"])   # convenient alias for push signal_id

        # ── 0. Corporate-action (split) guard ────────────────────────────────
        # Before ANY price-vs-level check: if a confirmed split moved this name to a
        # new scale, rescale the stored levels and skip this cycle — never let the
        # scale jump book a phantom stop (the KLAC 10:1 phantom -90% class of bug).
        try:
            _split_factor = _confirm_split_factor(sig, _current_price(ticker))
            if _split_factor and abs(_split_factor - 1.0) > 1e-9:
                _apply_split_adjustment(sb, sig, _split_factor)
                continue
        except Exception:
            pass

        # ── 1. Scalp time limit (30 min, any time) ───────────────────────────
        if strategy == "scalping" and age_mins >= SCALP_MAX_HOLD_MINS:
            price = _current_price(ticker)
            logger.info(f"[monitor] {ticker} scalp time limit ({age_mins:.0f} min) — closing")
            _close_signal(sb, sig["id"], "time_limit",
                          current_price=price,
                          entry_price=float(sig.get("entry_price") or 0),
                          direction=direction, ticker=ticker)
            _log_event(sb, sig["id"], "time_limit",
                       price=price,
                       note=f"30-min scalp window expired — position exited at ${price:.2f}" if price else
                            "30-min scalp window closed — position exited")
            try:
                _push_scalp_expired(ticker, direction, signal_id=sig_id_str)
            except Exception as _pe:
                logger.warning(f"[monitor] scalp-expired push failed for {sig_id_str}: {_pe}")
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
                        _push_eod_warning(ticker, direction, pnl_pct, signal_id=sig_id_str)
            except Exception:
                pass
            # Do NOT continue — still run normal checks below

        # ── 2b. Smart EOD close (3:30–4:05 PM) ───────────────────────────────
        #
        # 3:30 PM  Tier 1 — cut LOSERS only when momentum is also adverse
        #                    (winners and breakeven signals keep running)
        # 3:50 PM  Tier 2 — close all non-breakeven positions
        #                    (T1 hit + still in profit → let it ride to 3:55)
        # 3:55 PM  Tier 3 — absolute force-close everything remaining
        #
        # This replaces the old dumb 3:30 PM blanket close that killed
        # winning trades mid-power-hour.
        # A DAILY-timeframe signal is a multi-day swing by definition — never force it
        # flat at the bell even if its strategy label looks intraday (HOOD post-mortem:
        # swings must carry overnight; the May 28–29 +14% gap happened overnight).
        _is_daily_tf = str(sig.get("timeframe") or "").strip().lower() in ("1day", "1d", "d", "daily")
        if near_close and strategy in INTRADAY_STRATEGIES and not _is_daily_tf:
            price   = _current_price(ticker)
            entry   = float(sig.get("entry_price") or 0)
            is_long = direction == "LONG"
            pnl_pct: float | None = None
            if price and entry:
                raw     = ((price - entry) / entry * 100)
                pnl_pct = raw if is_long else -raw

            now_et_local = _now_et()
            eod_min      = now_et_local.hour * 60 + now_et_local.minute
            should_close = False

            if eod_min >= 15 * 60 + 55:          # ≥ 3:55 PM — no exceptions
                should_close = True
                logger.info(f"[monitor] {ticker} [{strategy}] 3:55 PM absolute force-close")

            elif eod_min >= 15 * 60 + 50:        # 3:50–3:54 PM — close losers not at breakeven
                # Only trim if the signal is BOTH (a) hasn't hit T1 yet (stop not at breakeven)
                # AND (b) is currently losing. A signal that's profitable but hasn't hit T1
                # yet (e.g. QCOM heading toward target) should ride to the 3:55 force-close
                # rather than be cut prematurely — it would have been a winner.
                at_breakeven = abs(float(sig.get("stop_loss") or 0) - entry) < 0.01
                in_profit    = pnl_pct is not None and pnl_pct > 0
                if not at_breakeven and not in_profit:
                    should_close = True
                    logger.info(
                        f"[monitor] {ticker} [{strategy}] 3:50 PM trim "
                        f"pnl={pnl_pct:.1f}% at_be={at_breakeven} — loss, no T1"
                    )

            elif eod_min >= 15 * 60 + 30:        # 3:30–3:49 PM — cut losers only
                if pnl_pct is not None and pnl_pct < -0.5:
                    book_now, _ = _momentum_check(ticker, direction)
                    if book_now or pnl_pct < -1.5:
                        should_close = True
                        logger.info(
                            f"[monitor] {ticker} [{strategy}] 3:30 PM loser cut "
                            f"pnl={pnl_pct:.1f}%"
                        )

            if should_close:
                pnl_str = (
                    f"+{pnl_pct:.1f}%" if pnl_pct and pnl_pct > 0
                    else f"{pnl_pct:.1f}%" if pnl_pct is not None
                    else ""
                )
                _close_signal(sb, sig["id"], "market_close",
                              current_price=price, entry_price=entry, direction=direction, ticker=ticker)
                _log_event(sb, sig["id"], "market_close", price=price,
                           note=f"Market closing — {direction} exited at ${price:.2f} {pnl_str}".strip()
                                if price else f"Market closing — {direction} position force-closed")
                try:
                    _push_market_close(ticker, direction, strategy, pnl_pct,
                                       signal_id=sig_id_str, created_at=sig.get("created_at"))
                except Exception as _pe:
                    logger.warning(f"[monitor] market-close push failed for {sig_id_str}: {_pe}")
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

        # Bad-print guard: a single out-of-tape SIP print must not drive status,
        # near-stop alerts, or the backstop close (the GLD 2026-06-03 near-stop
        # logged $417.54 while the 1-min high was $411.84). Clamp gross outliers
        # to the recent 1-min range — a real move that large appears in the bars,
        # so it is NOT clamped.
        price = _alpaca.sane_close_price(ticker, price) or price

        is_long  = direction == "LONG"
        pnl_pct  = ((price - entry) / entry * 100) if is_long else ((entry - price) / entry * 100)

        # ── 2b. Stop / target BACKSTOP close (tick-independent) ───────────────
        # The real-time exit (stream._check_rt_levels) is the fast path, but it
        # only fires for tickers in the worker's live trade subscription. An
        # active signal on a non-subscribed ticker (not a core name / current
        # mover) would otherwise NEVER have its stop or target enforced — the
        # trade hangs open even after the level is crossed (MSTR LONG sat past
        # its trailed stop at +1.3% and never booked, 2026-05-29). This 5-min
        # pass already holds a fresh real-time price, so enforce here as a
        # guaranteed backstop. Mirrors _check_rt_levels:
        #   T2 crossed → close (full target)
        #   SL crossed → close (win OR loss — result is by realized P&L, so a
        #                trailed stop above entry books the locked profit)
        # T1 → breakeven is handled by section 4 below; price between T1 and T2
        # falls through untouched.
        try:
            t2_cross = (price >= t2) if is_long else (price <= t2)
            sl_cross = (price <= sl) if is_long else (price >= sl)
            if t2_cross or sl_cross:
                # CONFIRM the cross before closing. A single bad/out-of-sequence
                # SIP last-trade print must not book a fake stop at a price the
                # tape never printed (2026-06-03 phantom-stop incident). Confirm
                # against recent 1-min bars or a fresh 2nd read; if unconfirmed,
                # skip the close this pass (re-checks next pass).
                _level = t2 if t2_cross else sl
                if not _alpaca.confirm_level_cross(ticker, _level, is_long,
                                                   "target" if t2_cross else "stop"):
                    logger.warning(
                        f"[monitor] {ticker} {'T2' if t2_cross else 'SL'} cross @ "
                        f"{price:.2f} NOT confirmed by bars/2nd-read — skipping "
                        f"(likely bad print; level={_level:.2f})"
                    )
                else:
                    # Record the exit at the LEVEL, never the overshoot print, so
                    # realized P&L reflects the actual stop/target (a stop at -3.9%
                    # books -3.9%, not -11.8%).
                    exit_px  = _level
                    pnl_exit = ((exit_px - entry) / entry * 100) if is_long else ((entry - exit_px) / entry * 100)
                    won      = pnl_exit >= 0
                    _close_signal(sb, sig["id"], "target_hit" if t2_cross else "stop_hit",
                                  current_price=exit_px, entry_price=entry, direction=direction)
                    if t2_cross:
                        note = f"🎯 Target hit @ ${exit_px:.2f} (+{pnl_exit:.1f}%)"
                    elif won:
                        note = f"✅ Stop reached in profit @ ${exit_px:.2f} (+{pnl_exit:.1f}%) — locked the gain"
                    else:
                        note = f"🔴 Stop hit @ ${exit_px:.2f} ({pnl_exit:.1f}%) — stopped out"
                    _log_event(sb, sig["id"], "closed_win" if won else "closed_loss", price=exit_px, note=note)
                    _STATUS_CACHE.pop(sig["id"], None)
                    _SWING_PEAK.pop(sig["id"], None)
                    logger.info(f"[monitor] {ticker} BACKSTOP close "
                                f"({'T2' if t2_cross else 'SL'}) @ {exit_px:.2f} ({pnl_exit:+.1f}%) "
                                f"[trigger {price:.2f}]")
                    continue
        except Exception as e:
            logger.debug(f"[monitor] backstop close error for {ticker}: {e}")

        # ── 3. Status change tracking + event logging ─────────────────────────
        try:
            new_status = _derive_status(price, entry, t1, sl, direction)
            old_status = _STATUS_CACHE.get(sig["id"])

            if new_status != old_status:
                _STATUS_CACHE[sig["id"]] = new_status
                _log_status_event(sb, sig["id"], new_status, price,
                                  direction=direction,
                                  extra=f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%",
                                  stop_loss=sl)
                # Push notification for important transitions only
                push_statuses = {"near_stop", "strong_profit", "at_target"}
                if new_status in push_statuses:
                    try:
                        _push_status_change(ticker, new_status, pnl_pct,
                                            direction=direction, signal_id=sig_id_str)
                    except Exception as _pe:
                        logger.warning(f"[monitor] status-change push failed for {sig_id_str}: {_pe}")
                logger.info(
                    f"[monitor] {ticker} status: {old_status} → {new_status} "
                    f"price={price:.2f} pnl={pnl_pct:+.2f}%"
                )
        except Exception as e:
            logger.debug(f"[monitor] Status tracking error for {ticker}: {e}")

        # ── 3b. Trend-ride: let a confirmed-green SWING run (HOOD post-mortem) ──
        # When a swing is green AND holding above a RISING 20-day MA, stop managing it
        # like a day-trade: trail the hard stop UNDER the recent daily swing low (ratchet
        # up only) and SKIP the early exits below (T1→BE tighten, peak-trail, intelligent
        # exit, near-expiry book, structure_reversal). Exit the ride only on a DECISIVE
        # daily CLOSE back through the 20-MA — intraday wicks never exit (HOOD 06-10 wicked
        # to $84 but CLOSED $86, then ran to $110). The hard stop + T1/T2 BACKSTOP above
        # still fire. Gated by TREND_RIDE_ENABLED; tagged (score_breakdown.trend_ride).
        _det_src = (sig.get("score_breakdown") or {}).get("detector_source")
        if (trend_ride.enabled() and trend_ride.is_swing(sig)
                and _det_src != "EMA_RECLAIM"):   # EMA_RECLAIM has its own 15m ride logic below
            try:
                _ctx = trend_ride.daily_context(ticker)
                _tr  = trend_ride.evaluate(sig, price, _ctx) if _ctx else None
                if _tr and _tr["break_exit"]:
                    logger.info(f"[monitor] {ticker} trend-ride EXIT — daily close {_tr['last_close']:.2f} "
                                f"{'<' if is_long else '>'} 20-MA {_tr['ma20']:.2f}")
                    _close_signal(sb, sig["id"], "trend_break",
                                  current_price=price, entry_price=entry, direction=direction, ticker=ticker)
                    _log_event(sb, sig["id"], "trend_break", price=price,
                               note=(f"📉 Daily close {_tr['last_close']:.2f} crossed back through the 20-MA "
                                     f"{_tr['ma20']:.2f} — trend-ride exit @ ${price:.2f} "
                                     f"({'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%)"))
                    _set_trend_ride_flag(sb, sig, False)
                    _STATUS_CACHE.pop(sig["id"], None)
                    _SWING_PEAK.pop(sig["id"], None)
                    continue
                if _tr and _tr["active"]:
                    _set_trend_ride_flag(sb, sig, True)
                    _new_sl  = _tr["trail_sl"]
                    _ratchet = (_new_sl > sl + 0.01) if is_long else (_new_sl < sl - 0.01)
                    if _ratchet:
                        _update_sl(sb, sig["id"], _new_sl, sig=sig)
                        _log_event(sb, sig["id"], "be_move", price=price,
                                   note=(f"🏄 Trend-ride: stop → ${_new_sl:.2f} under the daily swing low "
                                         f"(riding the rising 20-MA ${_tr['ma20']:.2f})"))
                        logger.info(f"[monitor] {ticker} trend-ride trail → {_new_sl:.2f} "
                                    f"(20-MA {_tr['ma20']:.2f}, pnl {pnl_pct:+.1f}%)")
                    continue   # ride on — skip the early-exit machinery below
                # No longer riding (lost green, or never cleared the MA) — clear any stale
                # flag and fall through to normal management.
                if _tr and _tr["was_riding"]:
                    _set_trend_ride_flag(sb, sig, False)
            except Exception as _e:
                logger.debug(f"[monitor] trend_ride error for {ticker}: {_e}")

        # ── 4. T1 hit → move SL to breakeven ─────────────────────────────────
        try:
            t1_hit = (is_long and price >= t1) or (not is_long and price <= t1)
            t2_hit = (is_long and price >= t2) or (not is_long and price <= t2)
            # Only if the stop is still BELOW breakeven — never loosen a stop the
            # pre-T1 peak trail (4b) may have already pushed into profit.
            sl_worse_than_be = (is_long and sl < entry - 0.01) or (not is_long and sl > entry + 0.01)
            if t1_hit and not t2_hit and sl_worse_than_be:
                pct = abs(price - entry) / entry * 100
                logger.info(f"[monitor] {ticker} T1 hit @ {price:.2f} — moving SL to breakeven")
                _update_sl(sb, sig["id"], entry, sig=sig)
                _log_event(sb, sig["id"], "t1_hit", price=price,
                           note=f"🎯 T1 hit @ ${price:.2f} (+{pct:.1f}%) — stop moved to breakeven ${entry:.2f}")
                try:
                    _push_t1_breakeven(ticker, direction, pct, signal_id=sig_id_str)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"[monitor] T1 check error for {ticker}: {e}")

        # ── 4b. Visible trailing stop — ride the trend after T1 ───────────────
        #
        # After T1 hits, the stop sits at breakeven (entry). That's safe but
        # leaves the displayed stop far from price as the trade runs — and gives
        # back the ENTIRE post-T1 gain on a reversal. This trails the VISIBLE
        # stop_loss up to lock in profit, so:
        #   • the card shows the real protected level (not stale breakeven)
        #   • the real-time tick SL check (_check_rt_levels) closes at the
        #     trailed stop automatically — no separate invisible mechanism
        #
        # Trail level locks TRAIL_LOCK_FRAC of the gain from entry to peak.
        # Ratchets UP only (never loosens). Activates once peak is a minimum
        # distance above T1 so we don't trail on noise right at T1.
        #
        # Example (SMCI): entry 36.90, peak 39.79 →
        #   trail = 36.90 + 0.70*(39.79-36.90) = $38.92  (locks +5.5%)
        try:
            t2_hit  = (is_long and price >= t2) or (not is_long and price <= t2)
            if not t2_hit:
                sig_id = sig["id"]
                # Track peak (best price seen on the trade)
                if sig_id not in _SWING_PEAK:
                    _SWING_PEAK[sig_id] = price
                elif is_long:
                    _SWING_PEAK[sig_id] = max(_SWING_PEAK[sig_id], price)
                else:
                    _SWING_PEAK[sig_id] = min(_SWING_PEAK[sig_id], price)
                peak = _SWING_PEAK[sig_id]

                # Activate trailing/intelligent-exit once T1 is hit OR peak has
                # covered _TRAIL_ACTIVATE_FRAC of the way to T1 (pre-T1 protection
                # for runners that stall before tagging the target — DVN).
                _denom   = (t1 - entry) if is_long else (entry - t1)
                progress = (((peak - entry) if is_long else (entry - peak)) / _denom) if _denom else 0.0
                t1_hit   = (is_long and price >= t1) or (not is_long and price <= t1)

                # ── Pre-T1 ABSOLUTE-PROFIT lock ───────────────────────────────
                # Far-target swings (breakdown shorts: T1 ≈ -10%+) would otherwise
                # give back a +2-3% gain while waiting for the 60%-to-T1 trail. Once
                # up _BE_PROFIT_PCT, ratchet the stop via the peak formula (floored
                # at breakeven) so the gain is protected. Tightens only (never
                # loosens, never closes). Direction-aware.
                if (not t1_hit) and progress < _TRAIL_ACTIVATE_FRAC \
                   and pnl_pct is not None and pnl_pct >= _BE_PROFIT_PCT:
                    # Confirmed breakdown/breakout: between +2% and +3% just guard
                    # at BREAKEVEN so a normal bounce can't stop us out near flat
                    # (CME). Only once genuinely profitable do we switch to the
                    # tight peak-trail. Other strategies tight-trail from +2%.
                    _tight_thresh = _LOCK_TIGHT_PCT.get(strategy)
                    _go_tight = (_tight_thresh is None) or (pnl_pct >= _tight_thresh)
                    if _go_tight:
                        _tp = TRAIL_PEAK_PCT.get(strategy, TRAIL_DEFAULT_PCT)
                        # Volatility-aware band: a fixed % trail is too tight for a
                        # high-ATR name — MSTR (ATR≈7%) got knocked out by a normal
                        # ~2% bounce while it kept falling. Widen toward ~0.5×ATR,
                        # capped 3.5%, floored at the per-strategy %.
                        try:
                            _atr_used = (sig.get("score_breakdown") or {}).get("atr_used")
                            if _atr_used and entry:
                                _atr_pct = float(_atr_used) / float(entry)
                                _tp = max(_tp, min(0.5 * _atr_pct, 0.035))
                        except Exception:
                            pass
                        if is_long: _tr = max(entry, peak * (1 - _tp))
                        else:       _tr = min(entry, peak * (1 + _tp))
                        _note_lbl = "🔒 Early profit lock"
                    else:
                        # Breakeven only — give the confirmed move room to ride.
                        _tr = round(float(entry), 2)
                        _note_lbl = "🛡️ Stop to breakeven"
                    _ratchet = (_tr > sl + 0.01) if is_long else (_tr < sl - 0.01)
                    if _ratchet:
                        _tr = round(_tr, 2)
                        _locked = ((_tr - entry) / entry * 100) if is_long else ((entry - _tr) / entry * 100)
                        _update_sl(sb, sig["id"], _tr, sig=sig)
                        _log_event(sb, sig["id"], "be_move", price=price,
                                   note=(f"{_note_lbl} → ${_tr:.2f} "
                                         f"(up {pnl_pct:.1f}%, locks {'+' if _locked >= 0 else ''}{_locked:.1f}%)"))
                        logger.info(f"[monitor] {ticker} {_note_lbl} → {_tr:.2f} "
                                    f"(pnl {pnl_pct:+.1f}%, locks {_locked:+.1f}%)")
                # EMA_RECLAIM trend-reclaim signals ride the move on the 15m
                # 20-EMA and SKIP the convergence early-exit — the point is to
                # not cut these winners early (HOOD/CRWD trend days).
                #
                # CRITICAL (per chart review): the ride ends on a CONFIRMED 15m
                # CLOSE below the 20 EMA — NOT on a wick. On 5m/1m the candles
                # routinely dip below the 9 EMA and wick under the 20 EMA mid-bar
                # while the 15m bar still closes above; a stop sitting at the
                # 20 EMA would get wicked out and cut the winner. So:
                #   • trend exit = last completed 15m close beyond the 20 EMA
                #   • the visible/tick stop is kept BELOW the recent 15m swing
                #     lows (not at the 20 EMA) so intrabar wicks don't trip it
                _detector_src = ((sig.get("score_breakdown") or {}).get("detector_source") or "")
                _is_ema_reclaim = _detector_src == "EMA_RECLAIM"

                if _is_ema_reclaim and (t1_hit or progress >= _TRAIL_ACTIVATE_FRAC):
                    try:
                        df_t  = smc.fetch_candles(ticker, period="2d", interval="15m")
                        ema20 = float(df_t["close"].ewm(span=20, adjust=False).mean().iloc[-1])
                        last_close = float(df_t["close"].iloc[-1])
                        rng   = (df_t["high"] - df_t["low"]).tail(14).mean()
                        buf   = float(rng) * 0.25 if rng and rng > 0 else price * 0.003
                        in_profit = (price > entry) if is_long else (price < entry)

                        # ── Trend break: confirmed 15m close beyond the 20 EMA ──
                        trend_broken = (is_long and last_close < ema20) or \
                                       (not is_long and last_close > ema20)
                        if trend_broken and in_profit:
                            pnl_x = ((price - entry) / entry * 100) if is_long \
                                    else ((entry - price) / entry * 100)
                            logger.info(f"[monitor] {ticker} EMA_RECLAIM trend exit — 15m close "
                                        f"{last_close:.2f} {'<' if is_long else '>'} 20-EMA {ema20:.2f}")
                            _close_signal(sb, sig_id, "target_hit",
                                          current_price=price, entry_price=entry, direction=direction, ticker=ticker)
                            _log_event(sb, sig_id, "closed_win", price=price,
                                       note=(f"📉 20-EMA close lost @ ${price:.2f} (+{pnl_x:.1f}%) — "
                                             f"trend exit, rode the move"))
                            _SWING_PEAK.pop(sig_id, None)
                            _STATUS_CACHE.pop(sig_id, None)
                            try:
                                _push_early_book(ticker, direction, round(pnl_x, 1),
                                                 "20-EMA close lost — trend exit", signal_id=sig_id_str)
                            except Exception:
                                pass
                            continue

                        # ── Ratchet the visible stop below recent swing lows ──
                        # (kept well under the 20 EMA so wicks don't trip it)
                        swing = float(df_t["low"].tail(3).min()) if is_long \
                                else float(df_t["high"].tail(3).max())
                        if is_long:
                            trail = max(entry, swing - buf)     # floor at breakeven
                            ratchet_up = trail > sl + 0.01
                        else:
                            trail = min(entry, swing + buf)
                            ratchet_up = trail < sl - 0.01
                        if ratchet_up:
                            trail = round(trail, 2)
                            locked = ((trail - entry) / entry * 100) if is_long \
                                     else ((entry - trail) / entry * 100)
                            _update_sl(sb, sig_id, trail, sig=sig)
                            _log_event(sb, sig_id, "be_move", price=price,
                                       note=(f"📈 Trail → ${trail:.2f} (below 15m swing, "
                                             f"rides 20-EMA, locks +{locked:.1f}%)"))
                            logger.info(f"[monitor] {ticker} EMA_RECLAIM trail → {trail:.2f} "
                                        f"(swing-based, ema20 {ema20:.2f}, locks +{locked:.1f}%)")
                    except Exception as e:
                        logger.debug(f"[monitor] EMA trail error for {ticker}: {e}")

                elif t1_hit or progress >= _TRAIL_ACTIVATE_FRAC:
                    # ── Intelligent exit: close EARLY if multiple real-time
                    #    signals converge on a reversal (vs blindly riding to T2).
                    #    Requires convergence so a single indicator can't bail. ──
                    try:
                        df_exit = smc.fetch_candles(ticker, period="2d", interval="15m")
                        tape_sum = None
                        try:
                            from engine import trade_tape
                            tape_sum = trade_tape.get_summary(ticker) or trade_tape.get_summary_redis(ticker)
                        except Exception:
                            pass
                        decision = exit_intelligence.evaluate_exit(sig, price, df_exit, peak, tape_sum)
                        if decision["action"] == "close":
                            reasons_str = ", ".join(decision["reasons"][:3])
                            logger.info(f"[monitor] {ticker} INTELLIGENT EXIT score={decision['score']} "
                                        f"pnl={decision['pnl_pct']}% — {reasons_str}")
                            _close_signal(sb, sig_id, "target_hit",
                                          current_price=price, entry_price=entry, direction=direction, ticker=ticker)
                            _log_event(sb, sig_id, "closed_win", price=price,
                                       note=(f"🧠 Intelligent exit @ ${price:.2f} "
                                             f"(+{decision['pnl_pct']}%) — {reasons_str}"))
                            _SWING_PEAK.pop(sig_id, None)
                            _STATUS_CACHE.pop(sig_id, None)
                            try:
                                _push_early_book(ticker, direction, decision["pnl_pct"],
                                                 f"Intelligent exit — {reasons_str}", signal_id=sig_id_str)
                            except Exception:
                                pass
                            continue
                    except Exception as e:
                        logger.debug(f"[monitor] intelligent exit error for {ticker}: {e}")

                    # Peak trailing: lock a fraction of the peak gain, floored at
                    # breakeven, ratchets UP only. Loose per-strategy % so swings
                    # (2% below peak) aren't whipsawed.
                    trail_pct = TRAIL_PEAK_PCT.get(strategy, TRAIL_DEFAULT_PCT)
                    if is_long:
                        trail = max(entry, peak * (1 - trail_pct))   # floor at breakeven
                        ratchet_up = trail > sl + 0.01
                    else:
                        trail = min(entry, peak * (1 + trail_pct))
                        ratchet_up = trail < sl - 0.01
                    if ratchet_up:
                        trail = round(trail, 2)
                        locked = ((trail - entry) / entry * 100) if is_long \
                                 else ((entry - trail) / entry * 100)
                        _update_sl(sb, sig_id, trail, sig=sig)
                        _log_event(sb, sig_id, "be_move", price=price,
                                   note=(f"📈 Trailing stop → ${trail:.2f} "
                                         f"({trail_pct*100:.1f}% below peak ${peak:.2f}, "
                                         f"locks +{locked:.1f}%)"))
                        logger.info(f"[monitor] {ticker} trailing stop → {trail:.2f} "
                                    f"(peak {peak:.2f}, locks +{locked:.1f}%)")
        except Exception as e:
            logger.debug(f"[monitor] Trailing stop error for {ticker}: {e}")

        # ── 4c. Near-expiry profit backstop (all strategies) ──────────────────
        # A position that chops in profit but never tags T1 (or trips the trail)
        # would otherwise expire near flat — DVN rode 2 days at +2-3% then expired.
        # Within the last _NEAR_EXPIRY_HRS of its max-hold, if still green, book it.
        try:
            from engine.runner import STRATEGY_MAX_HOLD_HOURS as _HOLD
            from datetime import datetime as _dtm, timezone as _tz
            _created = _dtm.fromisoformat(sig["created_at"].replace("Z", "+00:00"))
            _age_h   = (_dtm.now(_tz.utc) - _created).total_seconds() / 3600.0
            _hold_h  = _HOLD.get(strategy, 48.0)
            _t2_hit  = (is_long and price >= t2) or (not is_long and price <= t2)
            if (not _t2_hit) and _age_h >= (_hold_h - _NEAR_EXPIRY_HRS) and pnl_pct >= _NEAR_EXPIRY_MIN_PROFIT:
                logger.info(f"[monitor] {ticker} near-expiry profit book +{pnl_pct:.1f}% "
                            f"(age {_age_h:.1f}/{_hold_h}h)")
                _close_signal(sb, sig["id"], "target_hit",
                              current_price=price, entry_price=entry, direction=direction, ticker=ticker)
                _log_event(sb, sig["id"], "closed_win", price=price,
                           note=f"⏳ Near-expiry book @ ${price:.2f} (+{pnl_pct:.1f}%) — banked before expiry")
                _STATUS_CACHE.pop(sig["id"], None)
                _SWING_PEAK.pop(sig["id"], None)
                try:
                    _push_early_book(ticker, direction, pnl_pct, "Near-expiry profit book", signal_id=sig_id_str)
                except Exception:
                    pass
                continue
        except Exception as e:
            logger.debug(f"[monitor] near-expiry backstop error for {ticker}: {e}")

        # ── 4d. Time-stop: cut stale intraday trades that aren't working ──────
        # Winners resolve fast; losers drag to a full stop. If an intraday signal
        # hasn't made meaningful progress by _TIME_STOP_MINS, cut it (small loss/
        # scratch) instead of bleeding to the full stop.
        try:
            if strategy in _TIME_STOP_STRATEGIES:
                from datetime import datetime as _dts, timezone as _tzs
                _cd = _dts.fromisoformat(sig["created_at"].replace("Z", "+00:00"))
                _age_m = (_dts.now(_tzs.utc) - _cd).total_seconds() / 60.0
                _t2_hit = (is_long and price >= t2) or (not is_long and price <= t2)
                if (not _t2_hit) and _age_m >= _TIME_STOP_MINS and pnl_pct < _TIME_STOP_MIN_PROGRESS:
                    logger.info(f"[monitor] {ticker} TIME-STOP — no progress in {_age_m:.0f}m "
                                f"@ {price:.2f} ({pnl_pct:+.1f}%)")
                    _close_signal(sb, sig["id"], "time_limit",
                                  current_price=price, entry_price=entry, direction=direction, ticker=ticker)
                    _log_event(sb, sig["id"], "closed_loss" if pnl_pct < 0 else "closed_win", price=price,
                               note=(f"⏳ Time-stop @ ${price:.2f} ({pnl_pct:+.1f}%) — no progress "
                                     f"in {_age_m:.0f} min, cut to limit the drag"))
                    _STATUS_CACHE.pop(sig["id"], None)
                    _SWING_PEAK.pop(sig["id"], None)
                    continue
        except Exception as e:
            logger.debug(f"[monitor] time-stop error for {ticker}: {e}")

        # ── 5. Intelligent early booking (scalp + day_trade only) ─────────────
        #
        # SWING TRADES ARE INTENTIONALLY EXCLUDED from this block.
        #
        # Why: RSI/MACD readings are intraday noise for multi-day swing positions.
        # A SHORT swing will naturally show RSI oversold (momentum IN your favour)
        # and MACD patterns that look like reversals on the 5-min chart while the
        # daily/4h trend continues. Applying these diagnostics to swing trades
        # causes premature exits — exactly what happened with VLO SHORT:
        #   Engine closed at +1.79% citing "RSI oversold"
        #   Stock continued to +4–5% in the predicted direction.
        #
        # Swing exits are handled by:
        #   • T1/T2 RT level checker (stream.py — millisecond precision)
        #   • Swing trailing stop above (step 4b — protects gains after T1)
        #   • Structure reversal detection (step 6 below — CHoCH on 15m bars)
        #   • Smart EOD close is already exempt (swing not in INTRADAY_STRATEGIES)
        if strategy not in _SWING_LIKE_STRATEGIES:
            try:
                current_status = _STATUS_CACHE.get(sig["id"], "")
                # Pre-T1 only — once past T1, step 4b above owns the exit decision
                # (convergence engine + trailing stop). Avoids double-evaluation.
                past_t1 = (is_long and sl >= entry - 0.01) or (not is_long and sl <= entry + 0.01)
                should_assess = (current_status in ("building_profit", "strong_profit")
                                 and pnl_pct >= 1.0 and not past_t1)

                if should_assess:
                    # CONVERGENCE-BASED early book (same engine as step 4b). A
                    # single RSI/MACD reading no longer books profit — it needs
                    # 2+ real-time factors to agree (pressure >= 55). SHOP was
                    # closed below T1 on "RSI overbought (74)" ALONE, then ran on
                    # toward T2 — exactly the over-booking this prevents
                    # (fixed 2026-05-28). _momentum_check (single-indicator) is
                    # retained only for advisory pushes, not for closing.
                    df_exit = smc.fetch_candles(ticker, period="2d", interval="15m")
                    tape_sum = None
                    try:
                        from engine import trade_tape
                        tape_sum = trade_tape.get_summary(ticker) or trade_tape.get_summary_redis(ticker)
                    except Exception:
                        pass
                    decision = exit_intelligence.evaluate_exit(sig, price, df_exit, price, tape_sum)

                    # Time-based safety: a position stalling >3h with no progress
                    # still books to protect the gain (not a momentum call).
                    stalling = age_mins > 180 and current_status == "building_profit"

                    if decision["action"] == "close" or stalling:
                        reason = (", ".join(decision["reasons"][:3]) if decision["action"] == "close"
                                  else f"stalling {age_mins:.0f} min — protecting gains")
                        logger.info(f"[monitor] {ticker} EARLY BOOK (convergence) — {reason} pnl={pnl_pct:.1f}%")
                        _close_signal(sb, sig["id"], "target_hit",
                                      current_price=price, entry_price=entry, direction=direction, ticker=ticker)
                        _log_event(sb, sig["id"], "closed_win", price=price,
                                   note=f"💡 Profit booked @ ${price:.2f} (+{pnl_pct:.1f}%) — {reason}")
                        _STATUS_CACHE.pop(sig["id"], None)
                        try:
                            _push_early_book(ticker, direction, pnl_pct, reason, signal_id=sig_id_str)
                        except Exception:
                            pass
                        continue   # signal is now closed

                    # Convergence-tied breakeven: we're holding (pressure below the
                    # exit threshold) but there's MODERATE reversal pressure — so
                    # don't bail, just protect the unrealized gain by moving the
                    # stop to breakeven. Worst case becomes a scratch, not a loss,
                    # if the ride doesn't work out. Below this band we leave the
                    # stop alone so a clean trend can breathe. Once at breakeven,
                    # step 4b (trailing + convergence exit) takes over the ride.
                    elif decision["score"] >= _BE_PROTECT_SCORE and abs(sl - entry) > 0.01:
                        _update_sl(sb, sig["id"], round(entry, 2), sig=sig)
                        _log_event(sb, sig["id"], "be_move", price=price,
                                   note=(f"🛡 Stop → breakeven ${entry:.2f} — reversal pressure "
                                         f"{decision['score']}, protecting +{pnl_pct:.1f}% (riding, pre-T1)"))
                        logger.info(f"[monitor] {ticker} pre-T1 breakeven — pressure={decision['score']} "
                                    f"pnl={pnl_pct:.1f}%")
            except Exception as e:
                logger.debug(f"[monitor] Early booking check error for {ticker}: {e}")

        # ── 6. Structure reversal detection ───────────────────────────────────
        try:
            if _detect_structure_reversal(ticker, direction):
                logger.info(f"[monitor] {ticker} structure reversed — closing {direction} early")
                _close_signal(sb, sig["id"], "structure_reversal",
                              current_price=price, entry_price=entry, direction=direction, ticker=ticker)
                opposite = "bearish" if direction == "LONG" else "bullish"
                _log_event(sb, sig["id"], "reversal", price=price,
                           note=f"⚠️ {opposite.capitalize()} CHoCH detected — {direction} closed early @ ${price:.2f} ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%)")
                _STATUS_CACHE.pop(sig["id"], None)
                _SWING_PEAK.pop(sig["id"], None)
                try:
                    _push_reversal(ticker, direction, signal_id=sig_id_str)
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
        # MANUAL override: engine leaves admin-owned option signals untouched.
        if (sig.get("management_mode") or "engine") == "manual":
            continue
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
                    base = {
                        "status":        "closed",
                        "closed_reason": "target_hit" if result == "win" else "stop_hit",
                        "result":        result,
                        "closed_at":     now_utc.isoformat(),
                    }
                    # realized premium P&L% — needed to measure PUT/CALL expectancy.
                    # Try WITH the P&L cols; if they don't exist yet (migration
                    # supabase-option-result-pct.sql not run), still close the
                    # signal WITHOUT them — a close must never fail on a missing
                    # analytics column.
                    try:
                        sb.table("option_signals").update({
                            **base,
                            "result_pct": round(pct, 4),
                            "result_pnl": round(est_prem - entry_prem, 4),
                        }).eq("id", sig["id"]).execute()
                    except Exception as _col_e:
                        logger.warning(f"[monitor] option P&L cols missing for {ticker} "
                                       f"({_col_e}); closing without result_pct/pnl")
                        sb.table("option_signals").update(base).eq("id", sig["id"]).execute()
                    event_type = "closed_win" if result == "win" else "closed_loss"
                    event_note = (
                        f"Target hit — premium ${entry_prem:.2f}→${est_prem:.2f} (+{pct:.1f}%)"
                        if result == "win"
                        else f"Stop hit — premium ${entry_prem:.2f}→${est_prem:.2f} ({pct:.1f}%)"
                    )
                    _log_event(sb, sig["id"], event_type, price=price, note=event_note)
                    logger.info(f"[monitor] Option {ticker} {result} — premium {entry_prem:.2f}→{est_prem:.2f} ({pct:+.1f}%)")
                    try:
                        _ct = (sig.get("contract_type") or "").capitalize()
                        _k  = sig.get("strike_price")
                        _contract = (f"${float(_k):g} {_ct}".strip() if _k else (_ct or None))
                        _push_closed(ticker, direction, result, pct,
                                     created_at=sig.get("created_at"),
                                     is_option=True, contract=_contract,
                                     signal_id=str(sig.get("id")))
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
    # Skip on closed-market days. No prices can move, so no levels can
    # hit and no statuses can change. Belt-and-braces — the runner
    # maintenance job already short-circuits on holidays, but if anything
    # ever calls signal_monitor.run() directly this stays consistent.
    from engine.session_classifier import is_market_open_today
    if not is_market_open_today():
        logger.info("[monitor] Pass skipped — market closed today (holiday/weekend)")
        return

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
