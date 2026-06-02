"""
Cycle tracked-card generation — turnaround (LONG+CALL) + peak (SHORT+PUT).

The cycle PUSH alerts (Buy-Zone / Peak) already fire from runner's 5-min
breakout-watch sync. This module adds the TRADEABLE cards for those same names,
state-based and deduped, so a name that confirmed its turn overnight/pre-market
is captured at the RTH open (mirrors breakdown_alerts / breakout_alerts):

  • turnaroundStage == "buyzone" → turnaround_signals.generate (LONG + CALL)
  • peakStage       == "peak"    → peak_signals.generate       (SHORT + PUT)

Design (anti-spam), identical to breakdown/breakout:
  • Reuses the cached FULL quant scan (quant:scored:v1) — no extra fetches.
  • STATE-BASED, not transition-based: fires off the current confirmed stage so
    an overnight/pre-market confirm gets its card at the open.
  • Deduped via runner._has_active_signal → one card per ticker per episode.
  • HARD CAP per run (separately for turnaround + peak), ranked by score.
  • RTH-gated by the caller (options_scanner needs a live chain; you can't act
    on a long/short/option when the market is closed).
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.cycle_signals")

_MAX_TURN = 3   # cap tracked turnaround (long/call) cards per run
_MAX_PEAK = 3   # cap tracked peak (short/put) cards per run


def run(sb=None) -> dict:
    """Generate tracked cards for names CURRENTLY at a confirmed cycle turn.
    Best-effort — never raises into the scheduler."""
    from engine import cache, quant_score_service

    stats = {"scanned": 0, "turnaround_long": 0, "turnaround_call": 0,
             "peak_short": 0, "peak_put": 0}
    if sb is None:
        return stats

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
        logger.info("[cycle_signals] no scored universe available")
        return stats

    stats["scanned"] = len(scored)

    turn_now = [r for r in scored if r.get("turnaroundStage") == "buyzone"]
    peak_now = [r for r in scored if r.get("peakStage") == "peak"]
    turn_now.sort(key=lambda r: -float(r.get("turnaroundScore") or 0))
    peak_now.sort(key=lambda r: -float(r.get("peakScore") or 0))

    try:
        from engine import turnaround_signals, peak_signals, runner as _runner
    except Exception as e:
        logger.debug(f"[cycle_signals] import failed: {e}")
        return stats

    gen = 0
    for r in turn_now:
        if gen >= _MAX_TURN:
            break
        tk = (r.get("ticker") or "").upper()
        if not tk:
            continue
        try:
            if _runner._has_active_signal(sb, tk, "turnaround"):
                continue   # already tracked this episode
            res = turnaround_signals.generate(sb, r)
            if res.get("long") or res.get("call"):
                gen += 1
                if res.get("long"): stats["turnaround_long"] += 1
                if res.get("call"): stats["turnaround_call"] += 1
        except Exception as _e:
            logger.debug(f"[cycle_signals] turnaround gen failed for {tk}: {_e}")

    gen = 0
    for r in peak_now:
        if gen >= _MAX_PEAK:
            break
        tk = (r.get("ticker") or "").upper()
        if not tk:
            continue
        try:
            if _runner._has_active_signal(sb, tk, "peak"):
                continue
            res = peak_signals.generate(sb, r)
            if res.get("short") or res.get("put"):
                gen += 1
                if res.get("short"): stats["peak_short"] += 1
                if res.get("put"): stats["peak_put"] += 1
        except Exception as _e:
            logger.debug(f"[cycle_signals] peak gen failed for {tk}: {_e}")

    logger.info(f"[cycle_signals] done {stats}")
    return stats
