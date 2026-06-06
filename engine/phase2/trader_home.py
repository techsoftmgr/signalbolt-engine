"""
Module #7 — Trader Home Dashboard + AI Daily Briefing.

The single daily-use surface that aggregates the Phase 2 intelligence (threat
radar, watchlist intel, community intel, signal follow-up summary, market regime)
+ a plain-English AI Daily Briefing. Read-only; never raises; each sub-module is
included only if its own flag is on. Pure `briefing()` → unit-tested.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.phase2.trader_home")


def briefing(threat: dict | None, watch_ranked: list | None,
             comm_items: list | None, regime_type: str | None,
             active_signals: int | None = None) -> str:
    """Pure: the morning briefing text. Educational wording only."""
    p = ["Good morning."]
    if regime_type:
        p.append(f"Market regime is {regime_type.replace('_', ' ').title()}.")
    if threat and threat.get("level"):
        lvl = threat["level"]
        p.append(f"Threat level is {lvl} ({threat.get('threat_score')}/100)"
                 + (f" — {threat['reasons'][0]}." if threat.get("reasons") else "."))
    if watch_ranked:
        hot = [w for w in watch_ranked if w.get("priority", 0) >= 50]
        if hot:
            p.append(f"{len(hot)} watchlist name(s) need attention today — "
                     f"top: {hot[0]['ticker']} ({hot[0].get('why', '')}).")
        else:
            p.append("No watchlist names flag as urgent today.")
    if comm_items:
        real = [c for c in comm_items if c.get("verdict") == "REAL_MOMENTUM"]
        if real:
            p.append(f"Confirmed buzz: {real[0]['ticker']}.")
    if active_signals:
        p.append(f"You have {active_signals} active signal(s) to manage.")
    p.append("Manage risk accordingly. This is educational only, not financial advice.")
    return " ".join(p)


def _active_summary(sb):
    try:
        rows = (sb.table("signals").select("id,ticker,direction,status")
                .eq("status", "active").neq("strategy_type", "deep_value")
                .limit(500).execute().data) or []
        return {"active_count": len(rows)}
    except Exception:
        return {"active_count": None}


def dashboard(sb, tickers: list | None = None) -> dict:
    """Assemble the Trader Home dashboard. Never raises. Each block is gated by
    its own Phase 2 flag."""
    try:
        from engine.phase2 import flags, threat_radar, watchlist_intel, community_intel

        threat = threat_radar.compute() if flags.enabled("threat_radar") else None
        watch = (watchlist_intel.compute(sb, tickers or [])
                 if (flags.enabled("watchlist_intel") and tickers) else None)
        comm = community_intel.compute(sb, limit=8) if flags.enabled("community_intel") else None

        regime_type = None
        try:
            from engine import regime_detector
            regime_type = (regime_detector.detect() or {}).get("regime_type")
        except Exception:
            pass

        sig = _active_summary(sb)
        brief = (briefing(threat, (watch or {}).get("ranked"),
                          (comm or {}).get("items"), regime_type, sig.get("active_count"))
                 if flags.enabled("ai_briefing") else None)

        return {
            "enabled": True,
            "regime": regime_type,
            "threat_radar": threat,
            "watchlist_intel": watch,
            "community_intel": comm,
            "signals_summary": sig,
            "ai_briefing": brief,
        }
    except Exception as e:
        logger.error(f"[trader_home] failed: {e}")
        return {"enabled": True, "error": str(e)}
