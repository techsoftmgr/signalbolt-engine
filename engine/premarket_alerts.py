"""
Premarket disaster-gap alerts — notification-ONLY heads-up.

When an OPEN, overnight-held position has gapped hard AGAINST the signal before
the 9:30 open, push a "watch the open" warning so the holder isn't blindsided.

This is NOT a managed exit:
  • the engine never closes a position or records a win/loss on a premarket print
  • premarket prints are thin/wicky, options don't trade premarket, and the gap
    often reverses by the open — so acting on it would corrupt the track record
    and give unfillable exits. We only NOTIFY; the position is still exited the
    normal way (RTH tick checker / daily-close chandelier for TREND).

Scope: strategies actually held overnight —
  swing_trade · breakout · breakdown · TREND_MOMENTUM (via detector_source).
Intraday strategies (scalping / day_trade / options_flow) are flat by EOD, so
they have no overnight exposure and are skipped.

Timing: premarket window 8:00–9:30 AM ET ONLY — never earlier, to respect the
no-overnight-push rule. Per-signal-per-day dedup. Best-effort throughout.

Runs on a 15-min schedule from runner.py; the window/trading-day gate lives here.
"""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger("signalbolt.premarket_alerts")

ET = ZoneInfo("America/New_York")

_GAP_PCT   = 4.0           # adverse premarket move vs last RTH close to alert on
_DEDUP_TTL = 18 * 3600     # one alert per signal per premarket session
_OVERNIGHT_STRATEGIES = {"swing_trade", "breakout", "breakdown"}


def _in_window() -> bool:
    """True only during the 8:00–9:30 AM ET premarket window on a weekday."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 8 * 60 <= mins < 9 * 60 + 30


def _is_overnight(s: dict) -> bool:
    strat = s.get("strategy_type") or ""
    src   = (s.get("score_breakdown") or {}).get("detector_source")
    return strat in _OVERNIGHT_STRATEGIES or src == "TREND_MOMENTUM"


def run(sb=None) -> dict:
    """Push notification-only gap warnings for open overnight positions. Best-effort."""
    from engine import cache, push, alpaca_client
    from engine.session_classifier import is_market_open_today

    stats = {"held": 0, "checked": 0, "alerted": 0}
    if sb is None:
        return stats
    # Trading day + premarket window only (never overnight / after-hours / RTH).
    if not is_market_open_today() or not _in_window():
        return stats

    try:
        rows = (sb.table("signals").select("*").eq("status", "active").execute().data) or []
    except Exception as e:
        logger.error(f"[premarket_alerts] fetch failed: {e}")
        return stats

    held = [s for s in rows if _is_overnight(s)]
    stats["held"] = len(held)
    if not held:
        return stats

    tickers = sorted({(s.get("ticker") or "").upper() for s in held if s.get("ticker")})
    try:
        prices = alpaca_client.get_latest_prices(tickers) or {}
    except Exception as e:
        logger.error(f"[premarket_alerts] price fetch failed: {e}")
        return stats

    today = datetime.now(ET).strftime("%Y-%m-%d")
    ref_cache: dict[str, float] = {}   # last completed RTH daily close per ticker

    for s in held:
        tk = (s.get("ticker") or "").upper()
        px = prices.get(tk)
        if not tk or not px or px <= 0:
            continue

        # Reference = last completed RTH daily close (yesterday during premarket).
        # get_bars(1Day) is the reliable daily-close path (the snapshot daily_bar
        # object can be stale/forming); cache per ticker so we fetch once.
        ref = ref_cache.get(tk)
        if ref is None:
            try:
                df = alpaca_client.get_bars(tk, timeframe="1Day", days=5)
                ref = float(df["close"].iloc[-1]) if df is not None and len(df) else 0.0
            except Exception:
                ref = 0.0
            ref_cache[tk] = ref
        if not ref or ref <= 0:
            continue

        stats["checked"] += 1
        is_long = (s.get("direction") == "LONG")
        gap_pct = (px - ref) / ref * 100.0
        adverse = (gap_pct <= -_GAP_PCT) if is_long else (gap_pct >= _GAP_PCT)

        try:
            sl = float(s.get("stop_loss") or 0)
        except Exception:
            sl = 0.0
        through_stop = sl > 0 and ((is_long and px <= sl) or (not is_long and px >= sl))

        # Fire on a big adverse gap OR if premarket already breached the stop.
        if not (adverse or through_stop):
            continue

        dk = f"pm_gap:{s.get('id')}:{today}"
        try:
            if cache.kv.get_json(dk):
                continue
        except Exception:
            pass

        n = push.send_premarket_gap_alert(
            tk, s.get("direction", "LONG"), s.get("strategy_type") or "",
            round(gap_pct, 1), round(float(px), 2),
            round(sl, 2) if sl else None, through_stop,
            signal_id=str(s.get("id")),
        )
        try:
            cache.kv.set_json(dk, {"sent": n, "gap": round(gap_pct, 1)}, _DEDUP_TTL)
        except Exception:
            pass
        if n:
            stats["alerted"] += 1
            logger.info(f"[premarket_alerts] {tk} {s.get('direction')} gap {gap_pct:+.1f}% "
                        f"(through_stop={through_stop}) -> pushed {n}")

    logger.info(f"[premarket_alerts] done {stats}")
    return stats
