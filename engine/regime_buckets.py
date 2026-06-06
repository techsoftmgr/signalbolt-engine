"""
Regime buckets — collapse the 7 fine-grained regimes into 3 coarse buckets so
per-regime LEARNING (weights) and per-regime POLICY (detector sizing) can reach
a usable sample floor. Seven sparse cells rarely hit 30 trades; three buckets do.

Aligned with the regime→detector matrix:
  RISK_ON  = TRENDING_BULL, LOW_VOL          (trend-longs prime)
  NEUTRAL  = RANGING                          (chop — mean-reversion / quiet)
  RISK_OFF = HIGH_VOL, RISK_OFF, TRENDING_BEAR, PANIC  (shorts prime)

`bucket_of()` is pure. "ANY" / unknown → "ANY" so callers fall back to the
all-regime learned weights / default.
"""
from __future__ import annotations

RISK_ON  = "RISK_ON"
NEUTRAL  = "NEUTRAL"
RISK_OFF = "RISK_OFF"
ANY      = "ANY"

BUCKETS = (RISK_ON, NEUTRAL, RISK_OFF)

_MAP = {
    "TRENDING_BULL": RISK_ON,
    "LOW_VOL":       RISK_ON,
    "RANGING":       NEUTRAL,
    "HIGH_VOL":      RISK_OFF,
    "RISK_OFF":      RISK_OFF,
    "TRENDING_BEAR": RISK_OFF,
    "PANIC":         RISK_OFF,
}


def bucket_of(regime) -> str:
    """Map a fine regime → its bucket. Unknown / empty / 'ANY' → 'ANY'."""
    if not regime:
        return ANY
    r = str(regime).strip().upper()
    if r in BUCKETS or r == ANY:
        return r
    return _MAP.get(r, ANY)
