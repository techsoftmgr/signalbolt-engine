"""
Signal Advisor — Real-time dynamic hold/exit guidance for active signals.

Runs inside _check_rt_levels() on every live price tick (throttled to once
per 60 seconds per signal). Emits advisory push notifications + signal_events
entries when contextual conditions warrant user attention.

NO auto-close. Advice only — user decides.

Advisory checks (in priority order, first match wins per throttle window):
  1. Market close imminent (≥3:45 PM ET) + in profit + non-swing trade
     → "Consider closing before market close"
  2. Regime shifted to PANIC or HIGH_VOL (VIX spike > 15%)
     → "Market conditions changed — review your position"
  3. Day trade held > 5 hours (should close intraday)
     → "Day trade held 5+ hrs — consider exiting"
  4. Price 80–99% of the way to T1 (not yet hit)
     → "Approaching T1 — consider partial exit or tightening stop"
  5. T1 already hit + price ≥50% of the way from T1 → T2
     → "Strong momentum — T2 still reachable"

Per-type cooldowns prevent repeat spam for the same condition.
Global 60-second floor prevents any two advisories from the same signal
firing back to back.

Call evict(signal_id) when a signal closes to release memory.
"""

import logging
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger("signalbolt.signal_advisor")

ET = ZoneInfo("America/New_York")

# ── Global 60-second floor per signal ────────────────────────────────────────
# Prevents any two advisories from firing within 60 s of each other for the
# same signal, regardless of type.
_advisor_throttle: dict[str, float] = {}   # signal_id → last advice monotonic
_ADVISOR_THROTTLE_S = 60.0

# ── Per-type cooldowns per signal ─────────────────────────────────────────────
# On top of the 60-s floor, each advice type has its own minimum interval so
# the same condition can't re-fire immediately after the floor resets.
_advisor_type_last: dict[str, dict[str, float]] = {}  # signal_id → {type: monotonic}

_COOLDOWNS: dict[str, float] = {
    "market_close": 300.0,   # warn at most once per 5 min for close-imminent
    "regime_shift": 600.0,   # once per 10 min for regime changes
    "time_limit":   900.0,   # once per 15 min for time-limit warnings
    "near_t1":      300.0,   # once per 5 min for near-T1 hints
    "momentum_t2":  300.0,   # once per 5 min for T2 momentum hints
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cooldown_ok(sig_id: str, advice_type: str) -> bool:
    """Return True and record the timestamp if cooldown has elapsed."""
    now      = time.monotonic()
    type_map = _advisor_type_last.setdefault(sig_id, {})
    last     = type_map.get(advice_type, 0.0)
    if now - last >= _COOLDOWNS.get(advice_type, 300.0):
        type_map[advice_type] = now
        return True
    return False


def _send_advice(
    sig: dict,
    price: float,
    title: str,
    body: str,
    advice_type: str,
) -> None:
    """
    Write a signal_events row and fire a push notification.
    Both operations are best-effort — a failure here must never crash the caller.
    """
    try:
        import os
        from supabase import create_client as _sc
        from engine import push as _push

        key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
        sb  = _sc(os.environ["SUPABASE_URL"], key)

        sb.table("signal_events").insert({
            "signal_id":  sig["id"],
            "event_type": f"advisor_{advice_type}",
            "price":      price,
            "note":       body,
        }).execute()

        try:
            _push._send_raw(
                title=title,
                body=body,
                data={
                    "type":      f"advisor_{advice_type}",
                    "ticker":    sig["ticker"],
                    "signal_id": str(sig["id"]),
                },
            )
        except Exception:
            pass   # push failure is non-fatal

        logger.info(
            f"[advisor] {advice_type.upper()} {sig['ticker']} {sig['direction']} "
            f"@ ${price:.2f} — {title}"
        )

    except Exception as e:
        logger.debug(f"[advisor] _send_advice failed for {sig.get('id')}: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def check(sig: dict, price: float, regime: dict, session: dict) -> None:
    """
    Evaluate all advisory conditions for a live signal.

    Called from _check_rt_levels() on every throttled price tick.
    Returns immediately (no-op) if the per-signal 60-s floor has not elapsed.

    Parameters
    ----------
    sig     : signal dict from _rt_cache — must contain:
              id, ticker, direction, entry_price, stop_loss,
              target_one, target_two, strategy_type, created_at
    price   : current live trade price
    regime  : output of regime_detector.detect()
    session : output of session_classifier.classify()
    """
    sig_id = sig.get("id")
    if not sig_id:
        return

    # ── Global 60-s floor ────────────────────────────────────────────────────
    now = time.monotonic()
    if now - _advisor_throttle.get(sig_id, 0.0) < _ADVISOR_THROTTLE_S:
        return
    _advisor_throttle[sig_id] = now

    try:
        ticker    = sig["ticker"]
        direction = sig["direction"]
        is_long   = direction == "LONG"
        entry     = float(sig["entry_price"])
        sl        = float(sig["stop_loss"])
        t1        = float(sig["target_one"])
        t2        = float(sig["target_two"])
        strategy  = sig.get("strategy_type", "day_trade")

        pnl_pct = (
            (price - entry) / entry * 100
            if is_long
            else (entry - price) / entry * 100
        )

        # ── 1. Market close imminent + in profit ─────────────────────────────
        # 3:45 PM ET or later, any non-swing trade, position in profit.
        # Swing trades are meant to hold overnight — skip this check for them.
        if strategy != "swing_trade" and _cooldown_ok(sig_id, "market_close"):
            now_et = datetime.now(ET)
            if now_et.hour == 15 and now_et.minute >= 45 and pnl_pct > 0:
                _send_advice(
                    sig, price=price,
                    title=f"⏰ Market Closes Soon — {ticker}  {pnl_pct:+.1f}%",
                    body=(
                        f"{direction} is up {pnl_pct:.1f}% — market closes at 4 PM ET. "
                        f"Consider closing to lock in profit before the bell."
                    ),
                    advice_type="market_close",
                )
                return   # one advisory per throttle window to avoid overlap

        # ── 2. Regime shifted to danger zone ─────────────────────────────────
        # PANIC / HIGH_VOL regime or a sudden VIX spike (>15% change) are both
        # signals that the underlying market structure has deteriorated.
        if _cooldown_ok(sig_id, "regime_shift"):
            regime_type = regime.get("regime_type", "RANGING")
            vix_chg     = regime.get("vix_change_pct", 0.0)
            if regime_type in ("PANIC", "HIGH_VOL") or vix_chg > 15.0:
                spike_note = f" (VIX +{vix_chg:.0f}%)" if vix_chg > 15.0 else ""
                _send_advice(
                    sig, price=price,
                    title=f"⚡ Regime Alert — {ticker}",
                    body=(
                        f"Market shifted to {regime_type}{spike_note}. "
                        f"Your {direction} {strategy.replace('_', ' ')} may face increased risk. "
                        f"P&L: {pnl_pct:+.1f}% — review your position."
                    ),
                    advice_type="regime_shift",
                )
                return

        # ── 3. Day trade time limit (held > 5 hours) ─────────────────────────
        # An intraday trade held too long risks getting caught by the close.
        # Warn when the signal is 5+ hours old so the user has time to exit cleanly.
        if strategy == "day_trade" and _cooldown_ok(sig_id, "time_limit"):
            created_at = sig.get("created_at")
            if created_at:
                try:
                    if isinstance(created_at, str):
                        created_dt = datetime.fromisoformat(
                            created_at.replace("Z", "+00:00")
                        )
                    else:
                        created_dt = created_at
                    age_hours = (
                        datetime.now(timezone.utc) - created_dt
                    ).total_seconds() / 3600.0
                    if age_hours >= 5.0:
                        _send_advice(
                            sig, price=price,
                            title=f"⌛ Long Hold — {ticker}  {age_hours:.1f}h open",
                            body=(
                                f"This {direction} day trade has been open "
                                f"{age_hours:.1f} hours. "
                                f"Consider exiting before market close. "
                                f"P&L: {pnl_pct:+.1f}%."
                            ),
                            advice_type="time_limit",
                        )
                        return
                except Exception:
                    pass

        # ── 4. Approaching T1 (80–99% of the way) ────────────────────────────
        # Price is close to T1 but hasn't crossed it yet.
        # Useful for partial-exit or stop-tightening decisions.
        # Skip if T1 was already hit (stop == entry = breakeven).
        if _cooldown_ok(sig_id, "near_t1"):
            t1_already_hit = abs(sl - entry) < 0.01
            if not t1_already_hit:
                dist_entry_t1 = abs(t1 - entry)
                if dist_entry_t1 > 0:
                    dist_price_t1 = abs(t1 - price)
                    pct_toward_t1 = 1.0 - (dist_price_t1 / dist_entry_t1)
                    if 0.80 <= pct_toward_t1 < 1.0:
                        _send_advice(
                            sig, price=price,
                            title=(
                                f"🎯 Approaching T1 — {ticker} "
                                f" {pct_toward_t1 * 100:.0f}%"
                            ),
                            body=(
                                f"{direction} is {pct_toward_t1 * 100:.0f}% of the way "
                                f"to T1 (${t1:.2f}). P&L: {pnl_pct:+.1f}%. "
                                f"Consider partial exit or tightening your stop."
                            ),
                            advice_type="near_t1",
                        )
                        return

        # ── 5. T1 already hit + ≥50% toward T2 ──────────────────────────────
        # Stop is at breakeven, trade is riding to T2.
        # Confirm momentum when price clears the halfway point T1→T2.
        if _cooldown_ok(sig_id, "momentum_t2"):
            t1_already_hit = abs(sl - entry) < 0.01
            if t1_already_hit:
                dist_t1_t2 = abs(t2 - t1)
                if dist_t1_t2 > 0:
                    dist_price_t2 = abs(t2 - price)
                    pct_toward_t2 = 1.0 - (dist_price_t2 / dist_t1_t2)
                    if pct_toward_t2 >= 0.50:
                        _send_advice(
                            sig, price=price,
                            title=(
                                f"🚀 Riding to T2 — {ticker} "
                                f" {pct_toward_t2 * 100:.0f}% there"
                            ),
                            body=(
                                f"{direction} past T1, now {pct_toward_t2 * 100:.0f}% "
                                f"toward T2 (${t2:.2f}). Stop at breakeven — "
                                f"momentum is strong. P&L: {pnl_pct:+.1f}%."
                            ),
                            advice_type="momentum_t2",
                        )

    except Exception as e:
        logger.debug(f"[advisor] check() error for {sig_id}: {e}")


def evict(signal_id: str) -> None:
    """
    Clear all throttle state for a closed signal.
    Call this from every signal-close path so memory doesn't leak.
    """
    _advisor_throttle.pop(signal_id, None)
    _advisor_type_last.pop(signal_id, None)
    logger.debug(f"[advisor] Evicted state for signal {signal_id}")
