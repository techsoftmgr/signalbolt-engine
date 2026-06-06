"""
Regime-transition-aware exit — the DECISION BRAIN for "lock profit / cut when the
market dynamics flip against the position" (Layer 3 of the exit-learning roadmap).

PURE + tested + READY, but DELIBERATELY NOT WIRED anywhere live and NOT recording.
It plugs into two future places with a one-line call, when approved / when data
allows:
  • ENFORCE  — call assess() in the monitor/exit path; act on TIGHTEN/EXIT
               (needs user OK; would change live exits).
  • VALIDATE — use as a policy option in replay_backtest once regime_history has
               coverage over trades' lifetimes (forward-gated; no historical
               regime timeline yet).

Until then the evidence is reconstructable as an ANALYSIS: regime_history (market
timeline) × closed signals' give-back → "did an adverse regime flip mid-trade,
and how much did we hand back after?". No live recorder needed.

Adverse = the regime bucket moved AGAINST the position:
  LONG  worse as RISK_ON → NEUTRAL → RISK_OFF
  SHORT worse as RISK_OFF → NEUTRAL → RISK_ON
On an adverse flip: in profit → TIGHTEN (lock it); flat/red → EXIT (cut before
it worsens). Unknown ('ANY') buckets → HOLD (can't judge).
"""
from __future__ import annotations

from engine.regime_buckets import bucket_of, ANY

# Higher = friendlier to LONGs (risk-on). Used to detect directional adversity.
_RISK_SCORE = {"RISK_ON": 2, "NEUTRAL": 1, "RISK_OFF": 0}

# Master switch — when False, nothing in the engine acts on this module.
ENFORCE = False


def adverse_flip(direction: str, regime_at_entry, regime_now) -> bool:
    """Did the regime bucket move AGAINST the position since entry?"""
    b0, b1 = bucket_of(regime_at_entry), bucket_of(regime_now)
    if b0 == ANY or b1 == ANY or b0 == b1:
        return False
    is_long = (direction or "").upper() == "LONG"
    if is_long:
        return _RISK_SCORE[b1] < _RISK_SCORE[b0]      # got more risk-off
    return _RISK_SCORE[b1] > _RISK_SCORE[b0]           # got more risk-on (bad for a short)


def assess(direction: str, regime_at_entry, regime_now,
           unreal_pct: float | None) -> dict:
    """HOLD / TIGHTEN / EXIT recommendation on a regime transition. Pure."""
    b0, b1 = bucket_of(regime_at_entry), bucket_of(regime_now)
    if not adverse_flip(direction, regime_at_entry, regime_now):
        return {"action": "HOLD", "reason": "no adverse regime flip",
                "flip_from": b0, "flip_to": b1}
    u = None
    try:
        u = float(unreal_pct) if unreal_pct is not None else None
    except (TypeError, ValueError):
        u = None
    if u is not None and u > 0:
        return {"action": "TIGHTEN", "flip_from": b0, "flip_to": b1,
                "reason": f"regime {b0}→{b1} against a +{round(u, 1)}% position — lock the profit"}
    return {"action": "EXIT", "flip_from": b0, "flip_to": b1,
            "reason": f"regime {b0}→{b1} against position "
                      f"(unreal {round(u, 1) if u is not None else '?'}%) — cut before it worsens"}
