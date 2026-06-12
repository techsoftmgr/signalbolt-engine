"""
Momentum Surge detector — fires an EARLY long when a name is moving up fast on
heavy volume, caught while the move is YOUNG (~+5%) instead of after it has
already run +20% (the ROKU case: +20% on 5.4x volume never fired because every
trade gate correctly refused to chase an extended move).

Predictive family → REGIME-FREE: it fires via forming_signals.generate(), which
applies no regime / overextension gate (telemetry is logging-only). It's tagged
detector_source=MOMENTUM_SURGE so the detector scorecard measures its expectancy
— this is an explicit "catch it early" bet, kept or cut on the data, not vibes
(see [[no-detector-proliferation]]).

Why a +X% WINDOW (not just a floor): firing only between MIN_PCT and MAX_PCT is
the whole point — get in while the move is young, NOT after it's parabolic.

ENV (all optional):
  MOMENTUM_SURGE_ENABLED      default "true"  — master off-switch
  MOMENTUM_SURGE_MIN_PCT      default 5.0     — fire once the day's move clears +5%
  MOMENTUM_SURGE_MAX_PCT      default 12.0    — past this it's a chase; don't fire
  MOMENTUM_SURGE_MIN_RVOL     default 3.0     — heavy volume confirmation
  MOMENTUM_SURGE_MAX_PER_RUN  default 5
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.momentum_surge")

_MIN_PCT     = float(os.environ.get("MOMENTUM_SURGE_MIN_PCT", "5"))
_MAX_PCT     = float(os.environ.get("MOMENTUM_SURGE_MAX_PCT", "12"))
_MIN_RVOL    = float(os.environ.get("MOMENTUM_SURGE_MIN_RVOL", "3"))
_MAX_PER_RUN = int(os.environ.get("MOMENTUM_SURGE_MAX_PER_RUN", "5"))
_DEDUP_TTL   = 18 * 3600   # once per ticker per day


def _enabled() -> bool:
    return os.environ.get("MOMENTUM_SURGE_ENABLED", "true").lower() in ("1", "true", "yes", "on")


def _is_surge(r: dict) -> bool:
    """Young, heavy-volume up-move in an uptrend. The MAX_PCT ceiling is what makes
    this an EARLY entry rather than a chase of an already-extended name."""
    chg, rvol, px, ma = r.get("dayChangePct"), r.get("relativeVolume"), r.get("price"), r.get("ma20")
    if chg is None or rvol is None or not px or not ma:
        return False
    return (_MIN_PCT <= float(chg) <= _MAX_PCT) and float(rvol) >= _MIN_RVOL and float(px) > float(ma)


def run(sb=None) -> dict:
    """Scan the cached quant universe for young momentum surges and fire an early
    long for each (forming_signals, regime-free). Deduped once/ticker/day, capped."""
    stats = {"scanned": 0, "candidates": 0, "fired": 0}
    if not _enabled() or sb is None:
        return stats
    try:
        from engine import cache, quant_score_service, forming_signals
        scored = cache.kv.get_json(quant_score_service._SCORED_KEY) or []
    except Exception as e:
        logger.debug(f"[momentum_surge] universe load failed: {e}")
        return stats

    stats["scanned"] = len(scored)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cands = [r for r in scored if _is_surge(r)]
    # Strongest first: biggest move within the early window, then heaviest volume.
    cands.sort(key=lambda r: (float(r.get("dayChangePct") or 0), float(r.get("relativeVolume") or 0)), reverse=True)
    stats["candidates"] = len(cands)

    fired = 0
    for r in cands:
        if fired >= _MAX_PER_RUN:
            break
        tk = (r.get("ticker") or "").upper()
        if not tk:
            continue
        dk = f"surge_fired:{tk}:{today}"
        try:
            if cache.kv.get_json(dk):
                continue
        except Exception:
            pass
        try:
            if forming_signals.generate(sb, r, "surge").get("stock"):
                fired += 1
                try:
                    cache.kv.set_json(dk, {"fired": True}, _DEDUP_TTL)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"[momentum_surge] generate failed {tk}: {e}")

    stats["fired"] = fired
    if fired:
        logger.info(f"[momentum_surge] {stats}")
    return stats
