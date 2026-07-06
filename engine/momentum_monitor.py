"""
Momentum model — self-contained trade manager.

The systematic momentum/trend model owns its ENTIRE lifecycle: it fires its own
signals (runner._run_momentum_scan) and manages/closes them here, independent of
the generic signal_monitor (which is explicitly excluded from TREND_MOMENTUM).

Exit discipline = trend-following, NOT swing scalping:
  • CHANDELIER trailing stop (highest-high − 3×ATR(22)) — the proven, vol-
    adaptive trend-following stop. Ratchets up with the peak, never loosens.
  • SMA50-close structural backstop — confirms the trend regime has actually
    broken, not just a volatility spike.
  • DAILY-CLOSE discipline — exits only on a confirmed daily close beyond the
    stop, never an intraday wick (intraday noise routinely pierces these levels
    while the daily bar still closes inside the trend).
  • NO fixed profit target — let winners run; the chandelier captures the exit.

Runs once post-close (the daily bar is the event). Best-effort throughout.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from engine import alpaca_client
from engine.signal_monitor import _update_sl, _close_signal, _log_event

logger = logging.getLogger("signalbolt.momentum_monitor")

_ATR_PERIOD   = 22
_ATR_MULT     = 3.0     # base chandelier distance (let early trends breathe)
_HIGH_WINDOW  = 22      # rolling extreme for the chandelier anchor
_SMA_STRUCT   = 50      # structural trend backstop

# ── Profit-scaled trail (locks more of an outsized gain) ──────────────────────
# A pure 3×ATR chandelier on a high-vol name gives back a lot near the top —
# MRVL ran +52% (entry 207 → 314) with the stop ~35 pts (3×ATR≈$72) below price,
# locking only +17%. As the OPEN gain grows we step the ATR multiple DOWN, and
# never let the stop sit more than _GIVEBACK_CAP below the latest daily close.
# Tightens only the big winners; normal trends still ride on the full 3×ATR.
# Still 100% daily-close — no intraday / after-hours wick exits (thin AH prints
# would just shake a trend follower out, which is the opposite of the goal).
_PROFIT_TIERS      = ((50.0, 2.0), (25.0, 2.5))  # (open-gain% ≥, atr_mult), high→low
_GIVEBACK_CAP      = 0.20   # stop never sits >20% below the last close (long)…
_GIVEBACK_MIN_GAIN = 25.0   # …once the trade is up at least this %


def _atr_mult_for_gain(gain_pct: float) -> float:
    """Tighten the chandelier as the open gain grows: ride early, lock more once
    the move goes parabolic. Returns the ATR multiple for the current gain."""
    for thr, mult in _PROFIT_TIERS:     # ordered high → low
        if gain_pct >= thr:
            return mult
    return _ATR_MULT


def _atr(df: pd.DataFrame, period: int = _ATR_PERIOD) -> float:
    hi = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    cl = df["close"].values.astype(float)
    tr = np.maximum(hi[1:] - lo[1:],
                    np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])))
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) else 0.0
    return float(np.mean(tr[-period:]))


def _is_momentum(sig: dict) -> bool:
    return ((sig.get("score_breakdown") or {}).get("detector_source")) == "TREND_MOMENTUM"


def manage(sb) -> dict:
    """Manage all active TREND_MOMENTUM signals. Called post-close."""
    try:
        rows = (
            sb.table("signals").select("*")
              .eq("status", "active")
              .execute().data
        ) or []
    except Exception as e:
        logger.error(f"[momentum_monitor] fetch failed: {e}")
        return {"error": str(e)}

    mom = [r for r in rows if _is_momentum(r)]
    stats = {"checked": len(mom), "closed": 0, "trailed": 0, "skipped": 0}
    if not mom:
        return stats

    logger.info(f"[momentum_monitor] managing {len(mom)} TREND_MOMENTUM signal(s)")
    for sig in mom:
        try:
            # MANUAL override: admin owns it — engine does not trail/exit.
            if (sig.get("management_mode") or "engine") == "manual":
                stats["skipped"] += 1
                continue
            ticker    = sig["ticker"]
            is_long   = sig["direction"] == "LONG"
            entry     = float(sig["entry_price"])
            sl        = float(sig["stop_loss"])
            sig_id    = sig["id"]

            df = alpaca_client.get_bars(ticker, timeframe="1Day", days=90)
            if df is None or len(df) < _SMA_STRUCT + 2:
                stats["skipped"] += 1
                continue

            closes = df["close"].values.astype(float)
            last_close = float(closes[-1])
            atr   = _atr(df)
            sma50 = float(np.mean(closes[-_SMA_STRUCT:]))
            price = alpaca_client.get_latest_price(ticker) or last_close

            # Open gain off the DAILY close (never the live/AH price) so the
            # trail stays a pure daily-close decision. The multiple tightens as
            # the gain grows; the giveback floor caps how far below the close the
            # stop may sit once deeply profitable.
            gain_pct = ((last_close - entry) / entry * 100) if is_long \
                       else ((entry - last_close) / entry * 100)
            mult = _atr_mult_for_gain(gain_pct)
            _mtxt = f"{mult:g}×ATR"

            if is_long:
                roll_high  = float(np.max(df["high"].values.astype(float)[-_HIGH_WINDOW:]))
                chandelier = roll_high - mult * atr
                # Giveback floor: once up _GIVEBACK_MIN_GAIN, never let the stop
                # sit more than _GIVEBACK_CAP below the last close. Always below
                # last_close, so it can't force a spurious exit — only ratchets up.
                if gain_pct >= _GIVEBACK_MIN_GAIN:
                    chandelier = max(chandelier, last_close * (1 - _GIVEBACK_CAP))
                # Ratchet the stored stop UP toward the chandelier — never down.
                # The stop is a HARD FLOOR: the recomputed chandelier can WIDEN
                # (loosen) when the open gain shrinks — mult 2.5→3.0 — or ATR
                # expands, so exiting on the raw chandelier let a locked gain be
                # given back AND made the displayed stop a lie (LRCX: displayed
                # 372.49, raw chandelier re-widened to 348.74, close 353 → never
                # closed though "stopped"). Exit on the RATCHETED floor instead.
                eff_sl = round(max(sl, chandelier), 2)
                # Exit only on a CONFIRMED daily close below the ratcheted stop or
                # the structural SMA50 (trend regime break). Wicks are ignored.
                if last_close < eff_sl or last_close < sma50:
                    why = "trailing stop" if last_close < eff_sl else "SMA50 break"
                    _close_momentum(sb, sig_id, ticker, "LONG", entry, price, why)
                    stats["closed"] += 1
                    continue
                if eff_sl > sl + 0.01:
                    _update_sl(sb, sig_id, eff_sl, sig=sig)
                    _locked = (eff_sl - entry) / entry * 100
                    _log_event(sb, sig_id, "be_move", price=price,
                               note=(f"📈 Chandelier trail → ${eff_sl:.2f} "
                                     f"({_mtxt} below {roll_high:.2f}, locks "
                                     f"{'+' if _locked >= 0 else ''}{_locked:.0f}%, rides the trend)"))
                    stats["trailed"] += 1
            else:  # SHORT
                roll_low   = float(np.min(df["low"].values.astype(float)[-_HIGH_WINDOW:]))
                chandelier = roll_low + mult * atr
                if gain_pct >= _GIVEBACK_MIN_GAIN:
                    chandelier = min(chandelier, last_close * (1 + _GIVEBACK_CAP))
                # Ratchet DOWN only; exit on the ratcheted floor, not the raw
                # chandelier (mirror of the LONG fix — a loosening cover level must
                # not give back a locked short gain / contradict the displayed stop).
                eff_sl = round(min(sl, chandelier), 2)
                if last_close > eff_sl or last_close > sma50:
                    why = "trailing stop" if last_close > eff_sl else "SMA50 break"
                    _close_momentum(sb, sig_id, ticker, "SHORT", entry, price, why)
                    stats["closed"] += 1
                    continue
                if eff_sl < sl - 0.01:
                    _update_sl(sb, sig_id, eff_sl, sig=sig)
                    _locked = (entry - eff_sl) / entry * 100
                    _log_event(sb, sig_id, "be_move", price=price,
                               note=(f"📉 Chandelier trail → ${eff_sl:.2f} "
                                     f"({_mtxt} above {roll_low:.2f}, locks "
                                     f"{'+' if _locked >= 0 else ''}{_locked:.0f}%, rides the trend)"))
                    stats["trailed"] += 1
        except Exception as e:
            logger.debug(f"[momentum_monitor] {sig.get('ticker')} error: {e}")
            stats["skipped"] += 1

    logger.info(f"[momentum_monitor] done — {stats}")
    return stats


def _close_momentum(sb, sig_id, ticker, direction, entry, price, why: str) -> None:
    pnl = ((price - entry) / entry * 100) if direction == "LONG" \
          else ((entry - price) / entry * 100)
    won = pnl > 0
    _close_signal(sb, sig_id, "target_hit" if won else "stop_hit",
                  current_price=price, entry_price=entry, direction=direction, ticker=ticker)
    _log_event(sb, sig_id, "closed_win" if won else "closed_loss", price=price,
               note=(f"{'✅' if won else '🔴'} Trend exit ({why}) @ ${price:.2f} "
                     f"({'+' if pnl >= 0 else ''}{pnl:.1f}%) — rode the move"))
    logger.info(f"[momentum_monitor] {ticker} CLOSED via {why} @ ${price:.2f} ({pnl:+.1f}%)")


def stop_backstop(sb) -> dict:
    """
    LAST-RESORT safety net for TREND_MOMENTUM signals — closes any whose last
    COMPLETED daily close is beyond its stored stop_loss but that manage() failed
    to close (e.g. the primary job errored / didn't run — LRCX sat 34 days open).

    TREND_MOMENTUM is otherwise managed ONLY by manage() (generic signal_monitor
    skips it), so without this a manage() outage orphans the position with no stop.
    This runs on a SEPARATE schedule (see runner) so one job's failure can't
    disable the other. Daily-close based (respects the ignore-wicks design) and
    does NOTHING but enforce the stored stop — no trailing, no targets. The caller
    gates it to run only when the market is CLOSED, so closes[-1] is a settled bar.
    """
    stats = {"checked": 0, "closed": 0, "ok": 0}
    try:
        rows = (sb.table("signals").select("*").eq("status", "active").execute().data) or []
    except Exception as e:
        logger.error(f"[momentum_backstop] fetch failed: {e}")
        return {"error": str(e)}
    mom = [r for r in rows if _is_momentum(r) and (r.get("management_mode") or "engine") != "manual"]
    stats["checked"] = len(mom)
    for sig in mom:
        try:
            ticker = sig["ticker"]; is_long = sig["direction"] == "LONG"
            entry = float(sig["entry_price"]); sl = float(sig["stop_loss"]); sig_id = sig["id"]
            df = alpaca_client.get_bars(ticker, timeframe="1Day", days=6)
            if df is None or len(df) < 2:
                continue
            last_close = float(df["close"].values.astype(float)[-1])
            breached = (last_close < sl) if is_long else (last_close > sl)
            if not breached:
                stats["ok"] += 1
                continue
            # Re-validate against a sane close price so a single bad print can't
            # force a spurious backstop close (phantom-guard discipline).
            price = alpaca_client.sane_close_price(ticker, last_close) or last_close
            still = (price < sl) if is_long else (price > sl)
            if not still:
                stats["ok"] += 1
                continue
            logger.warning(f"[momentum_backstop] {ticker} daily close {last_close:.2f} beyond "
                           f"stored stop {sl:.2f} but still open — SAFETY CLOSE")
            _close_momentum(sb, sig_id, ticker, sig["direction"], entry, price,
                            "stop backstop (daily close beyond stored stop — manage() missed it)")
            stats["closed"] += 1
        except Exception as e:
            logger.debug(f"[momentum_backstop] {sig.get('ticker')} error: {e}")
    if stats["closed"]:
        logger.info(f"[momentum_backstop] done — {stats}")
    return stats
