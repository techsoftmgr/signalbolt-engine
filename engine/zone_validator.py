"""
Armed-zone counterfactual validator.

For zones that armed but never fired (the firing filters — retest / volume /
regime — skipped them), answer: *would a breakout entry have won?* This tells
us whether those filters are correctly avoiding losers or killing winners,
mirroring the gate-rejection validator.

Method per expired zone:
  1. Determine the breakout direction + entry level from the logged zone.
  2. Look at the 2h firing window: did price actually CROSS the entry level in
     that direction? If not → 'expired_no_trigger' (there was never a trade to
     judge), would_have_won stays NULL.
  3. If it did cross, simulate from the crossing using the live SL/TP engine,
     walking forward to the session close (intraday). Win = target before stop.

All DB work is best-effort. Runs post-close from the scheduler.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

from engine import alpaca_client, sl_tp_engine
from engine.gate_validator import _simulate

logger = logging.getLogger("signalbolt.zone_validator")

_ET = ZoneInfo("America/New_York")
_FIRING_WINDOW_H = 2.0      # zones can only fire within 2h of arming
_HOLD_H          = 8.0      # day_trade hold, capped to session close below


def _validate_one(row: dict) -> Optional[dict]:
    detector  = row.get("detector")
    armed_at  = row.get("armed_at")
    if not armed_at:
        return None
    try:
        t_arm = datetime.fromisoformat(armed_at.replace("Z", "+00:00"))
    except Exception:
        return None
    if t_arm.tzinfo is None:
        t_arm = t_arm.replace(tzinfo=timezone.utc)

    # Window must have fully elapsed (firing window + a little hold) before judging.
    if t_arm + timedelta(hours=_FIRING_WINDOW_H) > datetime.now(timezone.utc) - timedelta(minutes=15):
        return None

    # ── Resolve direction + breakout entry level from the zone shape ──────────
    level = row.get("armed_level")
    rng_hi, rng_lo = row.get("range_high"), row.get("range_low")
    direction = row.get("direction")
    if detector == "COMPRESSION":
        # Direction unknown at arm — decided by which side breaks (checked below).
        if rng_hi is None or rng_lo is None:
            return None
    elif level is None:
        return None
    elif direction not in ("LONG", "SHORT"):
        direction = "LONG"   # swing breakout default (armed_level = swing high)

    # ── Fetch bars: firing window (trigger detection) + hold (simulation) ─────
    bars = alpaca_client.get_bars(row.get("ticker"), timeframe="5Min", days=10)
    if bars is None or len(bars) < 30:
        return None
    t_end_window = t_arm + timedelta(hours=_FIRING_WINDOW_H)
    window = bars[(bars.index > t_arm) & (bars.index <= t_end_window)]
    if len(window) < 1:
        return None

    # ── Did a breakout actually trigger within the window? ────────────────────
    trig_idx = None
    if detector == "COMPRESSION":
        for ts, bar in window.iterrows():
            if float(bar["high"]) >= rng_hi:
                direction, level, trig_idx = "LONG", float(rng_hi), ts; break
            if float(bar["low"]) <= rng_lo:
                direction, level, trig_idx = "SHORT", float(rng_lo), ts; break
    else:
        for ts, bar in window.iterrows():
            crossed = (direction == "LONG"  and float(bar["high"]) >= level) or \
                      (direction == "SHORT" and float(bar["low"])  <= level)
            if crossed:
                trig_idx = ts; break

    if trig_idx is None:
        # Price never reached the breakout level — nothing to judge.
        return {"outcome": "expired_no_trigger", "would_have_won": None,
                "realized_pnl_pct": None}

    entry = float(level)
    context = bars[bars.index <= trig_idx]
    if len(context) < 25:
        return None

    # Cap the simulation to the session close (intraday never holds overnight).
    close_et = trig_idx.astimezone(_ET).replace(hour=16, minute=0, second=0, microsecond=0)
    t_sim_end = min(trig_idx + timedelta(hours=_HOLD_H), close_et.astimezone(timezone.utc))
    forward = bars[(bars.index > trig_idx) & (bars.index <= t_sim_end)]
    if len(forward) < 1:
        return None

    try:
        sltp = sl_tp_engine.calculate(
            direction=direction, entry=entry, df=context,
            regime={}, session={}, gamma={"available": False},
            strategy_type="day_trade", interval="5m",
        )
    except Exception:
        return None
    if not sltp.get("valid"):
        return {"outcome": "expired", "would_have_won": False, "realized_pnl_pct": 0.0,
                "sim_entry": round(entry, 4), "sim_stop": None, "sim_target": None}

    won, pnl = _simulate(direction=direction, entry=entry,
                         stop_loss=sltp["stop_loss"], target_one=sltp["target_one"],
                         forward_bars=forward)
    pnl_val = round(pnl, 4) if pnl is not None else 0.0
    return {
        "outcome":          "expired",
        "would_have_won":   bool(won) if won is not None else False,
        "realized_pnl_pct": pnl_val,
        "sim_entry":        round(entry, 4),
        "sim_stop":         round(sltp["stop_loss"], 4),
        "sim_target":       round(sltp["target_one"], 4),
    }


def validate_batch(sb, limit: int = 500) -> dict:
    """Judge unfired armed zones whose firing window has elapsed."""
    try:
        rows = (
            sb.table("armed_zone_history")
              .select("id, ticker, detector, direction, armed_level, range_high, "
                      "range_low, armed_at, outcome")
              .neq("outcome", "fired")
              .is_("would_have_won", "null")
              .order("armed_at", desc=True)
              .limit(max(1, min(limit, 2000)))
              .execute()
        ).data or []
    except Exception as e:
        logger.error(f"[zone_validator] fetch failed: {e}")
        return {"error": str(e), "processed": 0}

    stats = {"processed": 0, "wins": 0, "losses": 0, "no_trigger": 0, "skipped": 0}
    for row in rows:
        try:
            res = _validate_one(row)
            if res is None:
                stats["skipped"] += 1
                continue
            sb.table("armed_zone_history").update(res).eq("id", row["id"]).execute()
            stats["processed"] += 1
            if res.get("outcome") == "expired_no_trigger":
                stats["no_trigger"] += 1
            elif res.get("would_have_won"):
                stats["wins"] += 1
            else:
                stats["losses"] += 1
        except Exception as e:
            logger.debug(f"[zone_validator] row {row.get('id')} error: {e}")
            stats["skipped"] += 1
    logger.info(f"[zone_validator] batch done — {stats}")
    return stats
