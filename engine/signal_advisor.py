"""
Signal Advisor — Real-time dynamic hold/exit guidance for active signals.

Runs inside _check_rt_levels() on every live price tick (throttled to once
per 60 seconds per signal). Receives the rolling 10-minute price history so
it can detect fast market moves without any REST API calls.

Checks run in priority order — first match wins per throttle window:

  MARKET HEALTH (tick-level, price buffer, no REST):
  1. Rapid adverse move  — price dropped/spiked >0.8% against position in 60s
  2. Momentum exhaustion — price reversed >0.5% from peak after T1 was hit

  CONTEXTUAL (cached data, no REST):
  3. Market close imminent (≥3:45 PM ET) + in profit + non-swing
  4. Regime shifted to PANIC / HIGH_VOL (VIX spike >15%)
  5. Day trade held >5 hours
  6. Price 80–99% of the way to T1 (not yet hit)
  7. T1 hit + price ≥50% toward T2 (momentum confirmation)

All checks: push notification + signal_events entry.
No auto-close — user decides.
Throttle: 60-second global floor per signal + per-type cooldowns.

Call evict(signal_id) on every close path to release memory.
"""

import logging
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger("signalbolt.signal_advisor")

ET = ZoneInfo("America/New_York")

# ── Global 60-second floor per signal ────────────────────────────────────────
_advisor_throttle: dict[str, float] = {}
_ADVISOR_THROTTLE_S = 60.0

# ── Per-type cooldowns per signal ─────────────────────────────────────────────
_advisor_type_last: dict[str, dict[str, float]] = {}

_COOLDOWNS: dict[str, float] = {
    # Market health — more frequent because they're urgent
    "adverse_move":   120.0,   # warn at most every 2 min (fast moves repeat)
    "exhaustion":     300.0,   # once per 5 min (peak reversal)
    # Contextual
    "market_close":   300.0,
    "regime_shift":   600.0,
    "time_limit":     900.0,
    "near_t1":        300.0,
    "momentum_t2":    300.0,
}

# ── Peak price tracking per signal ────────────────────────────────────────────
# Tracks the best price seen for each active signal (highest for LONG,
# lowest for SHORT). Updated on every check() call regardless of throttle
# so the peak is always fresh when an exhaustion check runs.
_signal_peak: dict[str, float] = {}   # signal_id → best price seen


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cooldown_ok(sig_id: str, advice_type: str) -> bool:
    """Return True and stamp the timestamp if the cooldown has elapsed."""
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
    """Write a signal_events row and fire a push notification (best-effort)."""
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
            pass

        logger.info(
            f"[advisor] {advice_type.upper()} {sig['ticker']} {sig['direction']} "
            f"@ ${price:.2f} — {title}"
        )

    except Exception as e:
        logger.debug(f"[advisor] _send_advice failed for {sig.get('id')}: {e}")


def _update_peak(sig_id: str, price: float, is_long: bool) -> float:
    """
    Update and return the peak price for this signal.
    Peak = highest price for LONG (best profit level reached).
    Peak = lowest price for SHORT (best profit level reached).
    Initialised to current price on first call.
    """
    current_peak = _signal_peak.get(sig_id)
    if current_peak is None:
        _signal_peak[sig_id] = price
        return price
    if is_long:
        new_peak = max(current_peak, price)
    else:
        new_peak = min(current_peak, price)
    _signal_peak[sig_id] = new_peak
    return new_peak


# ── Market health checks (pure price-buffer logic, no REST) ───────────────────

def _check_adverse_move(
    sig: dict, price: float, price_history: list
) -> tuple[bool, str, str]:
    """
    Detect a rapid move against the signal direction in the last 60 seconds.

    Uses the 10-min price buffer — no REST call. Returns (fired, title, body).

    Threshold: 0.8% move against position in 60s = significant enough to warn.
    This catches:
      - Flash crashes against a LONG position
      - Sudden squeeze spikes against a SHORT position
      - News-driven reversals before RSI/MACD can react
    """
    if len(price_history) < 5:
        return False, "", ""

    now    = time.monotonic()
    cutoff = now - 60.0
    recent = [p for (p, t) in price_history if t >= cutoff]
    if len(recent) < 3:
        return False, "", ""

    is_long = sig["direction"] == "LONG"
    ticker  = sig["ticker"]

    if is_long:
        recent_high = max(recent)
        if recent_high <= 0:
            return False, "", ""
        drop_pct = (recent_high - price) / recent_high * 100
        if drop_pct >= 0.8:
            return (
                True,
                f"⚠️ Sudden Dump — {ticker}  -{drop_pct:.1f}% in 60s",
                f"LONG position: price dropped {drop_pct:.1f}% from ${recent_high:.2f} "
                f"to ${price:.2f} in under 60 seconds. Possible reversal — check your stop.",
            )
    else:  # SHORT
        recent_low = min(recent)
        if recent_low <= 0:
            return False, "", ""
        spike_pct = (price - recent_low) / recent_low * 100
        if spike_pct >= 0.8:
            return (
                True,
                f"⚠️ Sudden Spike — {ticker}  +{spike_pct:.1f}% in 60s",
                f"SHORT position: price spiked {spike_pct:.1f}% from ${recent_low:.2f} "
                f"to ${price:.2f} in under 60 seconds. Possible reversal — check your stop.",
            )

    return False, "", ""


def _check_exhaustion(
    sig: dict, price: float, peak: float
) -> tuple[bool, str, str]:
    """
    Detect momentum exhaustion after T1 is hit (stop at breakeven, riding to T2).

    Fires when price has reversed >0.5% from the peak seen since T1 was hit.
    This means the momentum that drove past T1 is fading — user should decide
    whether to hold to T2 or take the remaining profit now.
    """
    is_long = sig["direction"] == "LONG"
    ticker  = sig["ticker"]

    if is_long:
        if peak <= 0:
            return False, "", ""
        reversal_pct = (peak - price) / peak * 100
        if reversal_pct >= 0.5:
            return (
                True,
                f"📉 Momentum Fading — {ticker}  -{reversal_pct:.1f}% from peak",
                f"LONG trailing to T2: price pulled back {reversal_pct:.1f}% from peak ${peak:.2f}. "
                f"Momentum weakening — consider exiting or tightening stop.",
            )
    else:  # SHORT
        if peak <= 0:
            return False, "", ""
        reversal_pct = (price - peak) / peak * 100
        if reversal_pct >= 0.5:
            return (
                True,
                f"📉 Momentum Fading — {ticker}  +{reversal_pct:.1f}% from peak",
                f"SHORT trailing to T2: price bounced {reversal_pct:.1f}% from peak ${peak:.2f}. "
                f"Momentum weakening — consider exiting or tightening stop.",
            )

    return False, "", ""


# ── Public API ────────────────────────────────────────────────────────────────

def check(
    sig: dict,
    price: float,
    regime: dict,
    session: dict,
    price_history: list | None = None,
) -> None:
    """
    Evaluate all advisory conditions for a live signal and fire push
    notifications for any that trigger.

    Parameters
    ----------
    sig           : signal dict from _rt_cache — must contain:
                    id, ticker, direction, entry_price, stop_loss,
                    target_one, target_two, strategy_type, created_at
    price         : current live trade price
    regime        : output of regime_detector.detect()
    session       : output of session_classifier.classify()
    price_history : rolling 10-min list of (price, monotonic_ts) tuples
                    from _price_buffer in stream.py. If None, market-health
                    checks are skipped (no REST fallback needed).
    """
    sig_id = sig.get("id")
    if not sig_id:
        return

    try:
        is_long   = sig["direction"] == "LONG"
        entry     = float(sig["entry_price"])
        sl        = float(sig["stop_loss"])
        t1        = float(sig["target_one"])
        t2        = float(sig["target_two"])
        strategy  = sig.get("strategy_type", "day_trade")
        ticker    = sig["ticker"]
        direction = sig["direction"]

        pnl_pct = (
            (price - entry) / entry * 100
            if is_long
            else (entry - price) / entry * 100
        )

        # Always update peak — even when throttled so exhaustion check has
        # a fresh reference when the throttle floor finally allows it.
        peak = _update_peak(sig_id, price, is_long)

    except Exception as e:
        logger.debug(f"[advisor] pre-check error for {sig_id}: {e}")
        return

    # ── Global 60-second floor ────────────────────────────────────────────────
    now = time.monotonic()
    if now - _advisor_throttle.get(sig_id, 0.0) < _ADVISOR_THROTTLE_S:
        return
    _advisor_throttle[sig_id] = now

    # ═════════════════════════════════════════════════════════════════════════
    # TIER 1 — MARKET HEALTH  (tick-level, price buffer, no REST, highest urgency)
    # ═════════════════════════════════════════════════════════════════════════

    if price_history:
        # ── 1. Rapid adverse move ─────────────────────────────────────────────
        if _cooldown_ok(sig_id, "adverse_move"):
            fired, title, body = _check_adverse_move(sig, price, price_history)
            if fired:
                _send_advice(sig, price=price, title=title, body=body,
                             advice_type="adverse_move")
                return   # one advisory per throttle window

        # ── 2. Momentum exhaustion (only after T1 hit) ────────────────────────
        t1_already_hit = abs(sl - entry) < 0.01
        if t1_already_hit and _cooldown_ok(sig_id, "exhaustion"):
            fired, title, body = _check_exhaustion(sig, price, peak)
            if fired:
                _send_advice(sig, price=price, title=title, body=body,
                             advice_type="exhaustion")
                return

    # ═════════════════════════════════════════════════════════════════════════
    # TIER 2 — CONTEXTUAL  (cached regime/session, lower urgency)
    # ═════════════════════════════════════════════════════════════════════════

    # ── 3. Market close imminent + in profit ─────────────────────────────────
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
            return

    # ── 4. Regime shifted to danger zone ─────────────────────────────────────
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
                    f"P&L: {pnl_pct:+.1f}% — review position."
                ),
                advice_type="regime_shift",
            )
            return

    # ── 5. Day trade time limit (held >5 hours) ───────────────────────────────
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

    # ── 6. Approaching T1 (80–99% of the way, not yet hit) ───────────────────
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
                        title=f"🎯 Approaching T1 — {ticker}  {pct_toward_t1*100:.0f}%",
                        body=(
                            f"{direction} is {pct_toward_t1*100:.0f}% of the way to T1 "
                            f"(${t1:.2f}). P&L: {pnl_pct:+.1f}%. "
                            f"Consider partial exit or tightening stop."
                        ),
                        advice_type="near_t1",
                    )
                    return

    # ── 7. T1 hit + ≥50% toward T2 (momentum confirmation) ───────────────────
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
                        title=f"🚀 Riding to T2 — {ticker}  {pct_toward_t2*100:.0f}% there",
                        body=(
                            f"{direction} past T1, now {pct_toward_t2*100:.0f}% toward "
                            f"T2 (${t2:.2f}). Stop at breakeven — momentum strong. "
                            f"P&L: {pnl_pct:+.1f}%."
                        ),
                        advice_type="momentum_t2",
                    )


def evict(signal_id: str) -> None:
    """Clear all state for a closed signal to prevent memory leaks."""
    _advisor_throttle.pop(signal_id, None)
    _advisor_type_last.pop(signal_id, None)
    _signal_peak.pop(signal_id, None)
    logger.debug(f"[advisor] Evicted state for signal {signal_id}")
