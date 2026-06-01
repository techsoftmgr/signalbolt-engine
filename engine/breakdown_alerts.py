"""
Breakdown / heavy-selling alerts — universe-wide, two-stage.

Pushes ALL users (not watchlist-scoped) when a name in the scanned quant
universe starts breaking down, so a user can short / buy puts / exit even on a
ticker they don't already watch. Two stages:

  • EARLY     — a name LOSES its 20-day average on heavy down-volume
                (the earliest structural warning — "breakdown risk")
  • CONFIRMED — a name BREAKS its 20-day low on volume (the Breakdown bucket /
                setupType == "breakdown")

Design (anti-spam):
  • Reuses the cached FULL quant scan (quant:scored:v1) written by the dashboard
    precompute — NO extra Alpaca fetches.
  • Per-ticker state in Redis: we only fire on a genuine TRANSITION, never on
    every scan. The FIRST time we see a ticker we just SEED its baseline (so a
    deploy / cold cache can't fire a burst).
  • Per-ticker-per-stage-per-day dedup so an oscillating name can't spam.
  • HARD CAP per run (separately for early + confirmed), ranked by selling
    pressure, so a broad market selloff can't flood every user at once.
  • Pref-gated via push.send_breakdown_alert ('breakdown_alerts', default on).

Runs on a schedule (every ~15 min on trading days) from runner.py.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.breakdown_alerts")

_STATE_TTL   = 3 * 24 * 3600   # remember a ticker's state for 3 days
_DEDUP_TTL   = 36 * 3600       # one alert per ticker/stage per ~day
_MAX_EARLY   = 3               # cap EARLY pushes per run (anti-flood)
_MAX_CONFIRM = 3               # cap CONFIRMED pushes per run


def _heavy_down(r: dict) -> bool:
    """Heavy DOWN-volume today (the 'selling heavy' condition)."""
    return (r.get("volumeScore") or 0) >= 50 and (r.get("dayChangePct") or 0) < 0


def _pressure(r: dict) -> float:
    """Selling-pressure rank — mirrors the app's sellingPressure badge."""
    return max(float(r.get("breakdownScore") or 0.0), 0.85 * float(r.get("volumeScore") or 0.0))


def _state_of(r: dict) -> dict:
    px = r.get("price")
    ma = r.get("ma20")
    below = (px is not None and ma is not None and px < ma)
    return {
        "belowMA":   bool(below),
        "breakdown": r.get("setupType") == "breakdown",
    }


def run(sb=None) -> dict:
    """Scan the full quant universe, push on breakdown transitions. Best-effort."""
    from engine import cache, push, quant_score_service

    stats = {"scanned": 0, "early": 0, "confirmed": 0, "seeded": 0}

    # Reuse the cached full scan; if missing, trigger a dashboard build (which
    # also writes the scan) and read it back.
    scored = None
    try:
        scored = cache.kv.get_json(quant_score_service._SCORED_KEY)
    except Exception:
        scored = None
    if not scored:
        try:
            quant_score_service.get_quant_dashboard()
            scored = cache.kv.get_json(quant_score_service._SCORED_KEY)
        except Exception:
            scored = None
    if not scored:
        logger.info("[breakdown_alerts] no scored universe available")
        return stats

    stats["scanned"] = len(scored)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    early_cands: list[dict] = []
    confirm_cands: list[dict] = []

    for r in scored:
        tk = (r.get("ticker") or "").upper()
        if not tk:
            continue
        cur  = _state_of(r)
        prev = cache.kv.get_json(f"bd_state:{tk}")

        # Cold start: seed baseline, no alert (avoids a burst on first run).
        if prev is None:
            cache.kv.set_json(f"bd_state:{tk}", cur, _STATE_TTL)
            stats["seeded"] += 1
            continue

        # CONFIRMED: just entered the breakdown bucket (broke 20-day low on vol).
        if cur["breakdown"] and not prev.get("breakdown"):
            confirm_cands.append(r)
        # EARLY: just lost its 20-day average ON heavy down-volume, and not
        # already a confirmed breakdown (don't double-alert the same name).
        elif cur["belowMA"] and not prev.get("belowMA") and _heavy_down(r) and not cur["breakdown"]:
            early_cands.append(r)

        cache.kv.set_json(f"bd_state:{tk}", cur, _STATE_TTL)

    # Strongest selling pressure first, then cap.
    confirm_cands.sort(key=lambda r: -_pressure(r))
    early_cands.sort(key=lambda r: -_pressure(r))

    def _fire(r: dict, stage: str) -> int:
        tk = (r.get("ticker") or "").upper()
        dedup = f"bd_alert:{tk}:{stage}:{today}"
        if cache.kv.get_json(dedup):
            return 0
        rvol  = r.get("relativeVolume")
        extra = f"{rvol:.1f}x vol" if isinstance(rvol, (int, float)) else ""
        n = push.send_breakdown_alert(tk, stage, price=r.get("price"), extra=extra)
        cache.kv.set_json(dedup, {"sent": True, "n": n}, _DEDUP_TTL)
        if n:
            logger.info(f"[breakdown_alerts] {tk} {stage} -> pushed {n}")
        return n

    for r in confirm_cands[:_MAX_CONFIRM]:
        if _fire(r, "confirmed"):
            stats["confirmed"] += 1
    for r in early_cands[:_MAX_EARLY]:
        if _fire(r, "early"):
            stats["early"] += 1

    logger.info(f"[breakdown_alerts] done {stats}")
    return stats
