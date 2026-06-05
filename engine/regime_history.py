"""
Market-regime TIMELINE — append a row to `regime_history` ONLY when the regime
changes (write-on-change). The table then holds the day's transitions, e.g.
  05:00 pre TRENDING_BULL → 08:30 pre PANIC → 12:00 rth TRENDING_BULL → 14:00 rth RANGING
rather than a sample every tick.

We still EVALUATE the regime periodically (a small scheduled job calls
record_if_changed) because there's no "VIX crossed a threshold" event to hook —
but a DB row is written only on an actual regime_type/session flip (a handful per
day). An in-memory guard means repeat calls with the same regime never even hit
the DB.

Powers: exact intraday regime-at-fire, regime-during-hold studies, regime-aware
detector enablement/sizing, and a "market regime today" timeline.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger("signalbolt.regime_history")
ET = ZoneInfo("America/New_York")

# Per-process memo so unchanged regimes don't touch the DB at all.
_last: dict = {"regime_type": None, "session": None}


def session_now() -> str:
    """Coarse market session by ET wall-clock: pre | rth | post | closed."""
    et = datetime.now(ET)
    if et.weekday() >= 5:
        return "closed"
    mins = et.hour * 60 + et.minute
    if 4 * 60 <= mins < 9 * 60 + 30:    return "pre"
    if 9 * 60 + 30 <= mins <= 16 * 60:  return "rth"
    if 16 * 60 < mins <= 20 * 60:       return "post"
    return "closed"


def record_if_changed(sb, snap: dict | None = None) -> bool:
    """Append a regime_history row IFF regime_type or session changed since the
    last record. Returns True if a row was written. Never raises."""
    try:
        if snap is None:
            from engine import regime_detector
            snap = regime_detector.detect()
        rtype = (snap or {}).get("regime_type") or ""
        if not rtype:
            return False
        session = session_now()

        # 1) cheap in-memory short-circuit — no DB hit when nothing changed
        if rtype == _last["regime_type"] and session == _last["session"]:
            return False

        # 2) cross-process dedupe — another process (web/worker) may have already
        #    logged this same flip; only one row per transition.
        try:
            last = (sb.table("regime_history").select("regime_type,session")
                    .order("captured_at", desc=True).limit(1).execute().data)
            if last and last[0].get("regime_type") == rtype and last[0].get("session") == session:
                _last.update(regime_type=rtype, session=session)
                return False
        except Exception:
            pass

        row = {
            "regime_type":    rtype,
            "session":        session,
            "vix":            snap.get("vix"),
            "vix_change_pct": snap.get("vix_change_pct"),
            "adx":            snap.get("adx"),
            "above_200ma":    snap.get("above_200ma"),
            "spy_price":      snap.get("spy_price"),
            "fear_greed":     snap.get("fear_greed"),
            "blocked":        bool(snap.get("blocked")),
            # VIX (^VIX) is SPX-options-derived and thin outside RTH, so the
            # pre/post regime read is approximate — flag it for later analysis.
            "note":           "extended-hours VIX approximate" if session in ("pre", "post") else None,
        }
        sb.table("regime_history").insert(row).execute()
        _last.update(regime_type=rtype, session=session)
        logger.info(f"[regime_history] {session} → {rtype} (VIX {snap.get('vix')})")
        return True
    except Exception as e:
        logger.debug(f"[regime_history] record failed: {e}")
        return False


def recent(sb, hours: int = 48) -> list[dict]:
    """Regime transitions over the last N hours, oldest-first."""
    try:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        return (sb.table("regime_history").select("*")
                .gte("captured_at", since).order("captured_at").execute().data) or []
    except Exception as e:
        logger.debug(f"[regime_history] recent failed: {e}")
        return []
