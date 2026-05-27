"""
Entry Gate Rejection Validator (Chunk 2)
=========================================

Walks the `entry_gate_rejections` table and answers, for each row:
    "If we HAD fired this signal, would it have won or lost?"

Method:
  1. Fetch bars from Alpaca starting at the rejection's `created_at`
  2. Compute realistic SL/TP via sl_tp_engine on the bars *up to* that moment
  3. Walk forward bar-by-bar within the strategy's hold window:
       - LONG wins if high crosses target_one before low crosses stop_loss
       - LONG loses if low crosses stop_loss first
       - Time-out within hold window = inconclusive (counts as not-won)
  4. Backfill `would_have_won` (bool) and `realized_pnl_pct` on the row

Why this matters:
  - If validator says "gate rejected 60% winners" → gate is killing good trades, tune it down
  - If validator says "gate rejected 60% losers" → gate is doing its job, keep it
  - Single number that tells us if the gate is net-positive

Runs as:
  - On-demand via CLI: `python -m engine.gate_validator --limit=100`
  - Cron: nightly batch (default 200 rows) — see runner.py scheduler

Notes:
  - Skips rows where market was closed at created_at (weekends, holidays)
    OR insufficient forward bars are available (recent rejections from last hour).
    These remain NULL and get retried later.
  - Uses 5m bars for intraday strategies, 1h for swing — matches live engine cadence.
  - No regime/session/gamma context applied — uses neutral defaults. Slightly
    less accurate than live SL/TP but adequate for retrospective validation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

from engine import alpaca_client, sl_tp_engine

logger = logging.getLogger("signalbolt.gate_validator")


# ── Interval / hold-window per strategy ──────────────────────────────────────

_INTRADAY_STRATEGIES = {"scalping", "day_trade", "options_flow", "dark_pool", "gap_fill",
                        "vwap_reclaim", "pre_market", "earnings", "short_squeeze"}

# Match runner.STRATEGY_MAX_HOLD_HOURS so simulation matches live engine behavior
_MAX_HOLD_HOURS = {
    "scalping":       0.5,
    "day_trade":      8.0,
    "vwap_reclaim":   8.0,
    "gap_fill":       8.0,
    "pre_market":     8.0,
    "swing_trade":    240.0,
    "earnings":       48.0,
    "short_squeeze":  24.0,
    "position_trade": 720.0,
    "options_flow":   8.0,
    "dark_pool":      8.0,
}

# Bar interval used for forward-walk simulation
_SIM_INTERVAL = {
    "scalping":      "1Min",
    "day_trade":     "5Min",
    "options_flow":  "5Min",
    "dark_pool":     "5Min",
    "vwap_reclaim":  "5Min",
    "gap_fill":      "5Min",
    "pre_market":    "5Min",
    "earnings":      "15Min",
    "short_squeeze": "15Min",
    "swing_trade":   "1Hour",
    "position_trade":"1Hour",
}


# ── Outcome simulator ────────────────────────────────────────────────────────

def filter_rth(bars: pd.DataFrame) -> pd.DataFrame:
    """Public alias for the RTH filter so other modules (main.py, etc.) can use it."""
    return _filter_rth(bars)


def _filter_rth(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Drop bars outside US regular trading hours (9:30 AM – 4:00 PM ET,
    Mon-Fri). Eliminates a major source of validator-vs-reality drift:

      - After-hours bars (4 PM – 8 PM ET) have thin volume + wide spreads,
        so a "stop hit" on a 100-share AH trade isn't a realistic exit
      - Pre-market bars (4 AM – 9:30 AM ET) same problem
      - Weekend = no bars, but the time-walk window can extend through them

    By RTH-filtering, the simulated outcome matches what an actual trader
    using a regular market-hours order could realistically have achieved.
    """
    if bars is None or bars.empty:
        return bars
    try:
        # Bars come back from Alpaca with UTC tz; convert to ET for checking
        et = bars.index.tz_convert("America/New_York")
        is_weekday = et.dayofweek < 5
        minutes_from_open = (et.hour - 9) * 60 + et.minute - 30   # minutes past 9:30 ET
        rth_mask = is_weekday & (minutes_from_open >= 0) & (minutes_from_open < 390)  # 6.5h × 60
        return bars[rth_mask]
    except Exception as e:
        logger.debug(f"[validator] RTH filter failed, using all bars: {e}")
        return bars


def _simulate(
    direction: str,
    entry: float,
    stop_loss: float,
    target_one: float,
    forward_bars: pd.DataFrame,
) -> tuple[Optional[bool], Optional[float]]:
    """
    Walk forward bar-by-bar. Returns (won, realized_pnl_pct).

    won = True  if target_one hit before stop_loss
    won = False if stop_loss hit before target_one
    won = None  if neither hit within forward window (inconclusive)

    realized_pnl_pct is computed against entry price using the price level
    actually touched (target/stop) or last close (for inconclusive).

    Only RTH bars are considered (see _filter_rth) — after-hours / weekend
    fills aren't realistic exits for the average trader.
    """
    forward_bars = _filter_rth(forward_bars)
    if forward_bars is None or forward_bars.empty:
        return None, None

    for _, bar in forward_bars.iterrows():
        high = float(bar["high"])
        low  = float(bar["low"])

        if direction == "LONG":
            hit_target = high >= target_one
            hit_stop   = low  <= stop_loss
            # If a single bar contains both levels (volatile bar), we can't be
            # sure which hit first. Be conservative: assume stop hit first.
            if hit_target and hit_stop:
                return False, (stop_loss - entry) / entry * 100
            if hit_target:
                return True,  (target_one - entry) / entry * 100
            if hit_stop:
                return False, (stop_loss  - entry) / entry * 100
        else:  # SHORT
            hit_target = low  <= target_one
            hit_stop   = high >= stop_loss
            if hit_target and hit_stop:
                return False, (entry - stop_loss) / entry * 100
            if hit_target:
                return True,  (entry - target_one) / entry * 100
            if hit_stop:
                return False, (entry - stop_loss)  / entry * 100

    # Inconclusive — neither level hit within hold window
    last_close = float(forward_bars["close"].iloc[-1])
    if direction == "LONG":
        pnl_pct = (last_close - entry) / entry * 100
    else:
        pnl_pct = (entry - last_close) / entry * 100
    return None, pnl_pct


# ── Per-rejection processor ──────────────────────────────────────────────────

def _validate_one(row: dict) -> Optional[dict]:
    """
    Validate a single rejection row. Returns {would_have_won, realized_pnl_pct}
    on success, or None if it should be skipped (no data / market closed / etc.).
    """
    ticker        = row["ticker"]
    direction     = row["direction"]
    strategy_type = row["strategy_type"]
    entry         = float(row["price"]) if row.get("price") else None
    created_at    = row["created_at"]

    if not entry or not direction or not ticker:
        return None

    # Parse the rejection timestamp
    try:
        if isinstance(created_at, str):
            t_reject = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        else:
            t_reject = created_at
    except Exception:
        return None
    if t_reject.tzinfo is None:
        t_reject = t_reject.replace(tzinfo=timezone.utc)

    hold_hours = _MAX_HOLD_HOURS.get(strategy_type, 8.0)
    t_end      = t_reject + timedelta(hours=hold_hours)

    # If the hold window hasn't fully elapsed yet, skip — premature to judge
    if t_end > datetime.now(timezone.utc) - timedelta(minutes=15):
        return None

    # Fetch enough bars to (a) compute SL/TP context and (b) walk forward
    interval = _SIM_INTERVAL.get(strategy_type, "5Min")
    # We need ~5 days of context for ATR/ADR + forward bars covering hold_hours
    days_needed = max(5, int(hold_hours / 24) + 2)
    bars = alpaca_client.get_bars(ticker, timeframe=interval, days=days_needed + 7)
    if bars is None or len(bars) < 30:
        return None

    # Split into context (<= t_reject) and forward (> t_reject)
    context_bars = bars[bars.index <= t_reject]
    forward_bars = bars[(bars.index > t_reject) & (bars.index <= t_end)]

    if len(context_bars) < 25 or len(forward_bars) < 1:
        return None

    # Compute SL/TP using current engine logic. Neutral regime/session/gamma —
    # we lose a tiny bit of accuracy but it's good enough to validate the gate.
    try:
        sltp = sl_tp_engine.calculate(
            direction     = direction,
            entry         = entry,
            df            = context_bars,
            regime        = {},
            session       = {},
            gamma         = {"available": False},
            strategy_type = strategy_type,
            interval      = interval.replace("Min", "m").replace("Hour", "h"),
        )
    except Exception as e:
        logger.debug(f"[validator] sltp error {ticker}: {e}")
        return None

    if not sltp.get("valid"):
        # R:R below threshold — would have been blocked downstream too. Mark
        # as not-won (consistent with how the live engine would have handled it).
        return {"would_have_won": False, "realized_pnl_pct": 0.0}

    won, pnl = _simulate(
        direction   = direction,
        entry       = entry,
        stop_loss   = sltp["stop_loss"],
        target_one  = sltp["target_one"],
        forward_bars= forward_bars,
    )

    if won is None:
        # Inconclusive — count as "didn't win" but record the drift
        return {"would_have_won": False, "realized_pnl_pct": round(pnl, 4) if pnl is not None else 0.0}
    return {"would_have_won": won, "realized_pnl_pct": round(pnl, 4) if pnl is not None else 0.0}


# ── Public entry point ──────────────────────────────────────────────────────

def validate_batch(sb, limit: int = 200) -> dict:
    """
    Pull `limit` un-validated rejections and backfill their outcomes.
    Returns a summary dict for logging / reporting.
    """
    try:
        rows = (
            sb.table("entry_gate_rejections")
              .select("id, created_at, ticker, direction, strategy_type, price")
              .is_("would_have_won", "null")
              .order("created_at", desc=True)
              .limit(max(1, min(limit, 1000)))
              .execute()
        ).data or []
    except Exception as e:
        logger.error(f"[validator] fetch failed: {e}")
        return {"error": str(e), "processed": 0}

    stats = {"processed": 0, "wins": 0, "losses": 0, "skipped": 0, "errors": 0}
    for row in rows:
        try:
            result = _validate_one(row)
            if result is None:
                stats["skipped"] += 1
                continue
            sb.table("entry_gate_rejections").update(result).eq("id", row["id"]).execute()
            stats["processed"] += 1
            if result["would_have_won"]:
                stats["wins"] += 1
            else:
                stats["losses"] += 1
        except Exception as e:
            logger.debug(f"[validator] row {row.get('id')} error: {e}")
            stats["errors"] += 1

    # Headline number — what % of rejections would actually have lost?
    # High = gate is doing its job; low = gate is killing winners.
    judged = stats["wins"] + stats["losses"]
    stats["gate_correct_pct"] = round(stats["losses"] / judged * 100, 1) if judged > 0 else None
    logger.info(
        f"[validator] batch done — processed={stats['processed']} "
        f"wins={stats['wins']} losses={stats['losses']} "
        f"skipped={stats['skipped']} gate_correct={stats['gate_correct_pct']}%"
    )
    return stats


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, os
    from dotenv import load_dotenv
    from supabase import create_client

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])
    result = validate_batch(sb, limit=args.limit)
    print(result)
