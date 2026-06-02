"""
Breakout / unusual-buying alerts — universe-wide. Bullish mirror of
breakdown_alerts.

Pushes ALL users (not watchlist-scoped) when a name in the scanned quant universe
is breaking out or seeing unusual buying, so a user can go long / buy calls even
on a ticker they don't watch. Three event types:

  • EARLY breakout  — pressing its 20-day high on strong up-volume (setup forming)
  • CONFIRMED       — broke its 20-day high on volume → also generates LONG + CALL
                      cards (breakout_signals.generate)
  • ACCUMULATION    — heavy UP-volume (big buyers) without a structural break yet

Anti-spam, same design as breakdown_alerts: reuse the cached full quant scan,
per-ticker transition state in Redis (seed-on-first-sight), per-ticker-per-stage-
per-day dedup, hard per-run caps ranked by strength, pref-gated.

Runs every ~15 min during REGULAR market hours (gated in runner.py).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.breakout_alerts")

_STATE_TTL   = 3 * 24 * 3600
_DEDUP_TTL   = 36 * 3600
_MAX_CONFIRM = 3
_MAX_EARLY   = 3
_MAX_ACCUM   = 3


def _strength(r: dict) -> float:
    return max(float(r.get("breakoutScore") or 0.0),
               float(r.get("breakoutQuality") or 0.0),
               0.85 * float(r.get("volumeScore") or 0.0))


def _state_of(r: dict) -> dict:
    px = r.get("price"); hi = r.get("breakoutLevel"); ma = r.get("ma20")
    above_ma = px is not None and ma is not None and px > ma
    broke    = (r.get("setupType") == "breakout") or \
               (px is not None and hi is not None and px >= hi)
    dist     = r.get("distToBreakoutPct")             # negative = below the high
    near     = dist is not None and -2.5 <= dist < 0
    heavy_up = (r.get("volumeScore") or 0) >= 50 and (r.get("dayChangePct") or 0) > 0
    return {
        "breakout": bool(broke),
        "near":     bool(near and above_ma),
        "heavyup":  bool(heavy_up and above_ma),
    }


def run(sb=None) -> dict:
    """Scan the full quant universe, push on breakout/accumulation transitions."""
    from engine import cache, push, quant_score_service

    stats = {"scanned": 0, "confirmed": 0, "early": 0, "accum": 0, "seeded": 0,
             "long": 0, "call": 0}

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
        logger.info("[breakout_alerts] no scored universe available")
        return stats

    stats["scanned"] = len(scored)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    confirm_cands: list[dict] = []
    early_cands:   list[dict] = []
    accum_cands:   list[dict] = []

    for r in scored:
        tk = (r.get("ticker") or "").upper()
        if not tk:
            continue
        cur  = _state_of(r)
        prev = cache.kv.get_json(f"bo_state:{tk}")
        if prev is None:
            cache.kv.set_json(f"bo_state:{tk}", cur, _STATE_TTL)
            stats["seeded"] += 1
            continue

        if cur["breakout"] and not prev.get("breakout"):
            confirm_cands.append(r)
        elif (not cur["breakout"]) and cur["near"] and cur["heavyup"] and not prev.get("near"):
            early_cands.append(r)
        elif (not cur["breakout"]) and (not cur["near"]) and cur["heavyup"] and not prev.get("heavyup"):
            accum_cands.append(r)

        cache.kv.set_json(f"bo_state:{tk}", cur, _STATE_TTL)

    confirm_cands.sort(key=lambda r: -_strength(r))
    early_cands.sort(key=lambda r: -_strength(r))
    accum_cands.sort(key=lambda r: -float(r.get("volumeScore") or 0))

    def _extra(r: dict) -> str:
        rvol = r.get("relativeVolume")
        return f"{rvol:.1f}x vol" if isinstance(rvol, (int, float)) else ""

    def _deduped(key: str) -> bool:
        try:
            return bool(cache.kv.get_json(key))
        except Exception:
            return False

    def _mark(key: str, n: int) -> None:
        try:
            cache.kv.set_json(key, {"sent": True, "n": n}, _DEDUP_TTL)
        except Exception:
            pass

    for r in confirm_cands[:_MAX_CONFIRM]:
        tk = (r.get("ticker") or "").upper()
        dk = f"bo_alert:{tk}:confirmed:{today}"
        if not _deduped(dk):
            n = push.send_breakout_alert(tk, "confirmed", price=r.get("price"), extra=_extra(r))
            _mark(dk, n)
            if n:
                stats["confirmed"] += 1
        # Confirmed breakout → generate the tradeable LONG + CALL cards.
        try:
            from engine import breakout_signals
            res = breakout_signals.generate(sb, r)
            if res.get("long"): stats["long"] += 1
            if res.get("call"): stats["call"] += 1
        except Exception as _e:
            logger.debug(f"[breakout_alerts] signal gen failed: {_e}")

    for r in early_cands[:_MAX_EARLY]:
        tk = (r.get("ticker") or "").upper()
        dk = f"bo_alert:{tk}:early:{today}"
        if not _deduped(dk):
            n = push.send_breakout_alert(tk, "early", price=r.get("price"), extra=_extra(r))
            _mark(dk, n)
            if n:
                stats["early"] += 1

    for r in accum_cands[:_MAX_ACCUM]:
        tk = (r.get("ticker") or "").upper()
        dk = f"accum_alert:{tk}:{today}"
        if not _deduped(dk):
            n = push.send_accumulation_alert(tk, price=r.get("price"), extra=_extra(r))
            _mark(dk, n)
            if n:
                stats["accum"] += 1

    logger.info(f"[breakout_alerts] done {stats}")
    return stats
