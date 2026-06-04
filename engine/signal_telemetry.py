"""
Logging-only telemetry captured AT SIGNAL FIRE TIME, for later segmented
expectancy analysis (regime x instrument x concentration).

This NEVER gates or changes firing — every helper fails open and returns
partial/empty data on any error. It exists so the breakdown-quality study (and
any future detector study) has the fields it needs to slice realized outcomes:

  • market regime context  (regime_type column + VIX/ADX/SPY-vs-200MA)
  • concentration / correlation at fire (how many same-direction shorts were
    ALREADY open — the failure mode behind the 2026-06-04 breakdown blow-up,
    where 24 correlated shorts were held into one green reversal day)
  • sector (clustering)

Stored on the first-class `signals.regime_type` column PLUS a nested
`signals.score_breakdown["study"]` blob (JSONB — no migration needed).
Query later with e.g.  score_breakdown->'study'->>'regime_type'.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger("signalbolt.signal_telemetry")

# Regime is the same market-wide snapshot for every signal in a scan burst, and
# detect() does live VIX/SPY fetches — cache it briefly so a batch of breakdown
# fires doesn't trigger a fetch per signal.
_regime_cache: dict = {"data": None, "ts": 0.0}
_REGIME_TTL = 180.0   # seconds


def get_regime() -> dict:
    """Cached regime snapshot. Fails open to a neutral/empty dict."""
    now = time.monotonic()
    if _regime_cache["data"] is not None and (now - _regime_cache["ts"]) < _REGIME_TTL:
        return _regime_cache["data"]
    snap: dict
    try:
        from engine import regime_detector
        snap = regime_detector.detect()
    except Exception as e:
        logger.debug(f"[telemetry] regime detect failed: {e}")
        snap = {}
    _regime_cache["data"] = snap
    _regime_cache["ts"] = now
    return snap


def _count(sb, **eqs) -> int | None:
    """Exact COUNT of active `signals` rows matching the eq filters. Cheap
    (count is returned in the Content-Range header regardless of limit). Fails
    open to None so a counting hiccup never blocks a fire."""
    if sb is None:
        return None
    try:
        q = sb.table("signals").select("id", count="exact")
        for k, v in eqs.items():
            q = q.eq(k, v)
        return q.limit(1).execute().count
    except Exception as e:
        logger.debug(f"[telemetry] count {eqs} failed: {e}")
        return None


def capture(sb, ticker: str, direction: str, strategy_type: str) -> tuple[str, dict]:
    """Return (regime_type, study_blob) to record at fire time.

    `regime_type` populates the signals.regime_type column; `study_blob` is
    merged into score_breakdown["study"]. The concentration counts are taken
    BEFORE this signal is inserted, so they answer "how many were already open
    when this fired". Never raises."""
    study: dict = {}
    regime_type = ""
    direction = (direction or "").upper()

    # ── market regime context ──
    try:
        reg = get_regime()
        regime_type = reg.get("regime_type") or ""
        study["regime_type"]     = regime_type
        study["vix"]             = reg.get("vix")
        study["vix_change_pct"]  = reg.get("vix_change_pct")
        study["adx"]             = reg.get("adx")
        study["spy_above_200ma"] = reg.get("above_200ma")
    except Exception as e:
        logger.debug(f"[telemetry] regime capture failed: {e}")

    # ── sector (clustering / correlation) ──
    try:
        from engine import risk_manager
        study["sector"] = risk_manager.get_sector((ticker or "").upper())
    except Exception as e:
        logger.debug(f"[telemetry] sector failed: {e}")

    # ── concentration AT fire (the key new metric) ──
    if direction:
        # whole-book same-direction exposure …
        study["open_dir_total"] = _count(sb, status="active", direction=direction)
        # … and same-detector same-direction (this detector's own cluster)
        if strategy_type:
            study["open_strat"] = _count(
                sb, status="active", direction=direction, strategy_type=strategy_type)

    return regime_type, study
