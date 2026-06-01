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
_ATR_MULT     = 3.0     # chandelier distance
_HIGH_WINDOW  = 22      # rolling extreme for the chandelier anchor
_SMA_STRUCT   = 50      # structural trend backstop


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

            if is_long:
                roll_high  = float(np.max(df["high"].values.astype(float)[-_HIGH_WINDOW:]))
                chandelier = roll_high - _ATR_MULT * atr
                # Exit only on a CONFIRMED daily close below the stop or the
                # structural SMA50 (trend regime break). Wicks are ignored.
                if last_close < chandelier or last_close < sma50:
                    why = "chandelier" if last_close < chandelier else "SMA50 break"
                    _close_momentum(sb, sig_id, ticker, "LONG", entry, price, why)
                    stats["closed"] += 1
                    continue
                new_sl = round(max(sl, chandelier), 2)   # ratchet up only
                if new_sl > sl + 0.01:
                    _update_sl(sb, sig_id, new_sl, sig=sig)
                    _log_event(sb, sig_id, "be_move", price=price,
                               note=(f"📈 Chandelier trail → ${new_sl:.2f} "
                                     f"(3×ATR below {roll_high:.2f}, rides the trend)"))
                    stats["trailed"] += 1
            else:  # SHORT
                roll_low   = float(np.min(df["low"].values.astype(float)[-_HIGH_WINDOW:]))
                chandelier = roll_low + _ATR_MULT * atr
                if last_close > chandelier or last_close > sma50:
                    why = "chandelier" if last_close > chandelier else "SMA50 break"
                    _close_momentum(sb, sig_id, ticker, "SHORT", entry, price, why)
                    stats["closed"] += 1
                    continue
                new_sl = round(min(sl, chandelier), 2)
                if new_sl < sl - 0.01:
                    _update_sl(sb, sig_id, new_sl, sig=sig)
                    _log_event(sb, sig_id, "be_move", price=price,
                               note=(f"📉 Chandelier trail → ${new_sl:.2f} "
                                     f"(3×ATR above {roll_low:.2f}, rides the trend)"))
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
                  current_price=price, entry_price=entry, direction=direction)
    _log_event(sb, sig_id, "closed_win" if won else "closed_loss", price=price,
               note=(f"{'✅' if won else '🔴'} Trend exit ({why}) @ ${price:.2f} "
                     f"({'+' if pnl >= 0 else ''}{pnl:.1f}%) — rode the move"))
    logger.info(f"[momentum_monitor] {ticker} CLOSED via {why} @ ${price:.2f} ({pnl:+.1f}%)")
