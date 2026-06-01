"""
Watchlist state-change alerts.

Pushes a user when one of THEIR watched tickers changes situation:
  • enters a Buy Zone        (turnaround reversal confirmed)
  • starts Topping / Peak    (distribution risk)
  • breaks out / actionable  (watchStatus -> actionable)
  • loses its trend          (drops below the 20-day average)

Design (anti-spam):
  • Per-ticker state is cached in Redis. We only push on a genuine TRANSITION,
    never on every scan.
  • The FIRST time we ever see a ticker we just SEED its baseline state and send
    nothing — so a deploy / cold cache can't fire a burst of alerts.
  • Per-ticker-per-event-per-day dedup so an oscillating state can't spam.
  • Watchlist-scoped + pref-gated via push.send_watchlist_state_alert.

Runs on a schedule (every ~15 min on trading days) from runner.py.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.watchlist_alerts")

_STATE_TTL   = 3 * 24 * 3600   # remember a ticker's state for 3 days
_DEDUP_TTL   = 36 * 3600       # one alert per ticker/event per ~day
_MAX_TICKERS = 80              # bound per-run cost


def _state_of(q: dict, price) -> dict:
    ma    = q.get("ma20")
    above = (price is not None and ma is not None and price > ma)
    return {
        "turn":    q.get("turnaroundStage") or "none",
        "peak":    q.get("peakStage") or "none",
        "status":  q.get("watchStatus") or "",
        "aboveMA": bool(above),
    }


def _events(prev: dict, cur: dict) -> list[tuple[str, str, str]]:
    """(event_key, title, body) for each NEW alert-worthy transition prev -> cur."""
    out: list[tuple[str, str, str]] = []
    if cur["turn"] == "buyzone" and prev.get("turn") != "buyzone":
        out.append(("buyzone", "🟢 {t} — Buy Zone",
                    "{t} is showing a bottoming reversal. Tap for the game plan."))
    if cur["peak"] in ("watch", "peak") and (prev.get("peak") in (None, "none", "")):
        out.append(("topping", "🔻 {t} — Topping",
                    "{t} looks like it's peaking — consider trimming. Tap for details."))
    if cur["status"] == "actionable" and prev.get("status") != "actionable":
        out.append(("actionable", "⚡ {t} — Actionable",
                    "{t} just turned actionable (broke its level). Tap for the plan."))
    if (not cur["aboveMA"]) and prev.get("aboveMA") is True:
        out.append(("losttrend", "⚠️ {t} — Trend lost",
                    "{t} dropped below its 20-day average. Tap for details."))
    return out


def run(sb) -> dict:
    """Scan all watched tickers, push on state transitions. Best-effort."""
    from engine import cache, push, quant_score_service, alpaca_client, regime_detector

    stats = {"tickers": 0, "alerts": 0, "seeded": 0}
    try:
        rows = sb.table("watchlist").select("ticker").execute().data or []
    except Exception as e:
        logger.error(f"[watchlist_alerts] fetch watchlist failed: {e}")
        return stats

    tickers = list({(r.get("ticker") or "").upper() for r in rows if r.get("ticker")})[:_MAX_TICKERS]
    if not tickers:
        return stats
    stats["tickers"] = len(tickers)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        regime_type = (regime_detector.detect() or {}).get("regime_type")
    except Exception:
        regime_type = None
    try:
        spy_long = alpaca_client.get_bars("SPY", "1Day", days=400)
    except Exception:
        spy_long = None

    for tk in tickers:
        try:
            daily = alpaca_client.get_bars(tk, "1Day", days=400)
            if daily is None or len(daily) < 60:
                continue
            intraday = alpaca_client.get_bars(tk, "15Min", days=5)
            price    = alpaca_client.get_latest_price(tk)
            q = quant_score_service._score_ticker(
                tk, price, daily, intraday,
                daily_long_df=daily, regime_type=regime_type, spy_long_df=spy_long,
            )
            if not q:
                continue
            cur  = _state_of(q, price)
            prev = cache.kv.get_json(f"wl_state:{tk}")

            # Cold start: seed baseline, no alert (avoids a burst on first run).
            if prev is None:
                cache.kv.set_json(f"wl_state:{tk}", cur, _STATE_TTL)
                stats["seeded"] += 1
                continue

            for ev_key, title_tpl, body_tpl in _events(prev, cur):
                dedup = f"wl_alert:{tk}:{ev_key}:{today}"
                if cache.kv.get_json(dedup):
                    continue
                n = push.send_watchlist_state_alert(
                    tk, title_tpl.format(t=tk), body_tpl.format(t=tk), sb=sb,
                )
                cache.kv.set_json(dedup, {"sent": True, "n": n}, _DEDUP_TTL)
                if n:
                    stats["alerts"] += 1
                logger.info(f"[watchlist_alerts] {tk} {ev_key} -> pushed {n}")

            cache.kv.set_json(f"wl_state:{tk}", cur, _STATE_TTL)
        except Exception as e:
            logger.debug(f"[watchlist_alerts] {tk} error: {e}")

    logger.info(f"[watchlist_alerts] done {stats}")
    return stats
