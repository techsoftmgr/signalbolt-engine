"""
Phase 2 feature flags — the kill-switch layer.

Every Phase 2 endpoint checks a flag before doing anything; if off, it returns
{"enabled": false} and touches nothing. Read-only, zero-risk modules default ON
so they're usable out of the box; anything invasive (broker connections, writes)
defaults OFF until explicitly enabled.

Override any flag at runtime via env: PHASE2_<NAME>=true|false|1|0|on|off.
Example: PHASE2_PORTFOLIO_DOCTOR=true
"""
from __future__ import annotations

import os

# per-flag defaults — invasive modules OFF, read-only modules ON
_DEFAULTS: dict[str, bool] = {
    "threat_radar":     True,    # read-only market dashboard — safe
    "watchlist_intel":  False,
    "community_intel":  False,
    "position_coach":   False,
    "signal_followup":  False,
    "portfolio_doctor": False,   # broker connections — invasive, OFF
    "trader_home":      False,
    "ai_briefing":      False,
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def enabled(name: str) -> bool:
    """Is a Phase 2 module enabled? env override → per-flag default → False."""
    env = os.environ.get(f"PHASE2_{name.upper()}")
    if env is not None:
        v = env.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
    return _DEFAULTS.get(name, False)


def all_flags() -> dict:
    """Current state of every Phase 2 flag (for an admin/status view)."""
    return {k: enabled(k) for k in _DEFAULTS}
