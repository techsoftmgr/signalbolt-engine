"""
Churn/Absorption — resolution scorecard (#1) + multi-day coiling streak (#2).

Measure-first, NO firing change: does absorption predict the next move, and does longer coiling
mean a bigger move? Mirrors breakout_watch_history — we persist the OCCURRENCE (one row per
session × ticker that hit the Churn/Absorption screen, with its zone + coiling streak) and score
the FORWARD outcome on the fly from daily bars.

Zones (the directional read):
  • accumulation (near lows)  → expected to resolve UP   (buyers absorbing supply)
  • distribution (near highs) → expected to resolve DOWN (sellers feeding the top)
  • churn        (mid-range)  → no directional bias; we just track the size of the eventual move

snapshot_today(sb)  — run once/day AFTER the close: persist today's realized absorption set + streak.
score(sb, days)     — forward-return-by-zone (+ by streak bucket), judged HORIZON_DAYS out.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger("signalbolt.churn_history")

_TABLE        = "churn_history"
_ET           = ZoneInfo("America/New_York")
HORIZON_DAYS  = 5      # trading sessions after the observation that count as the "resolution"
_STREAK_GAP   = 4      # ≤ this many calendar days back = the previous trading session (coiling continues)


# ── Snapshot (persist today's absorption set + coiling streak) ───────────────
def snapshot_today(sb) -> dict:
    """Persist today's Churn/Absorption occurrences and carry the multi-day streak. Idempotent
    (upsert on session_date+ticker). Returns {snapshot, ...}; degrades to {} on any failure."""
    try:
        from engine.churn_service import compute_churn
        items = (compute_churn(force=True) or {}).get("items") or []
        if not items:
            return {"snapshot": 0}
        sd = datetime.now(_ET).date()
        tickers = [it["symbol"] for it in items]

        # Prior streaks: the latest churn_history row per ticker BEFORE today.
        prev: dict[str, dict] = {}
        try:
            rows = (sb.table(_TABLE).select("ticker,session_date,streak")
                    .in_("ticker", tickers).lt("session_date", sd.isoformat())
                    .order("session_date", desc=True).execute().data) or []
            for r in rows:
                prev.setdefault(r["ticker"], r)   # first seen = most recent (desc order)
        except Exception as e:
            logger.debug(f"[churn_history] prior-streak load failed: {e}")

        payload = []
        for it in items:
            tk = it["symbol"]
            streak = 1
            p = prev.get(tk)
            if p:
                try:
                    gap = (sd - datetime.fromisoformat(str(p["session_date"])).date()).days
                    if 0 < gap <= _STREAK_GAP:
                        streak = int(p.get("streak") or 1) + 1
                except Exception:
                    pass
            payload.append({
                "session_date": sd.isoformat(), "ticker": tk, "zone": it.get("zone"),
                "rel_vol": it.get("relVol"), "churn_score": it.get("churnScore"),
                "range_pos": it.get("rangePos"), "event": bool(it.get("event")),
                "obs_close": it.get("price"), "streak": streak,
            })
        sb.table(_TABLE).upsert(payload, on_conflict="session_date,ticker").execute()
        logger.info(f"[churn_history] snapshot {sd}: {len(payload)} names")
        return {"snapshot": len(payload), "date": sd.isoformat()}
    except Exception as e:
        logger.error(f"[churn_history] snapshot failed: {e}")
        return {}


# ── Forward outcome + aggregation (pure / testable) ──────────────────────────
def _forward_return(df, session_date, obs_close, horizon: int = HORIZON_DAYS):
    """Net % move over the `horizon` trading sessions AFTER session_date, from daily bars.
    Returns None when the horizon hasn't elapsed yet or the bar isn't found (not judgeable)."""
    try:
        # Daily bars are date-keyed; session_date was stored as that same calendar label, so match
        # on the bar date directly (Alpaca daily timestamps sit at the session date — same label).
        dates = list(df.index.date)
        if session_date not in dates:
            return None
        i = dates.index(session_date)
        j = i + horizon
        if j >= len(dates):
            return None                      # horizon not elapsed
        base = float(obs_close) if obs_close else float(df["close"].iloc[i])
        if base <= 0:
            return None
        fwd_close = float(df["close"].iloc[j])
        return (fwd_close / base - 1) * 100
    except Exception:
        return None


def _streak_bucket(s: int) -> str:
    return "1" if s <= 1 else "2" if s == 2 else "3+"


# zone → the direction that counts as a CORRECT resolution
_ZONE_DIR = {"accumulation": "up", "distribution": "down"}


def _aggregate(judged: list[dict]) -> dict:
    """judged = occurrence rows each with a numeric `fwd` (% move over the horizon)."""
    def _stats(rows: list[dict], expected: str | None) -> dict:
        n = len(rows)
        if not n:
            return {"n": 0, "avgFwdPct": 0.0, "upRate": 0.0, "hitRate": None, "avgAbsPct": 0.0}
        fwds = [r["fwd"] for r in rows]
        up = sum(1 for f in fwds if f > 0)
        if expected == "up":
            hit = up
        elif expected == "down":
            hit = sum(1 for f in fwds if f < 0)
        else:
            hit = None
        return {
            "n": n,
            "avgFwdPct": round(sum(fwds) / n, 2),
            "upRate": round(100 * up / n, 1),
            "hitRate": (round(100 * hit / n, 1) if hit is not None else None),
            "avgAbsPct": round(sum(abs(f) for f in fwds) / n, 2),
        }

    by_zone = {}
    for z in ("accumulation", "distribution", "churn"):
        by_zone[z] = _stats([r for r in judged if r.get("zone") == z], _ZONE_DIR.get(z))
    by_streak = {}
    for b in ("1", "2", "3+"):
        by_streak[b] = _stats([r for r in judged if _streak_bucket(int(r.get("streak") or 1)) == b], None)
    return {
        "n": len(judged),
        "horizonDays": HORIZON_DAYS,
        "byZone": by_zone,
        "byStreak": by_streak,
    }


# ── Scorecard (on-the-fly forward outcome) ───────────────────────────────────
def score(sb, days: int = 60) -> dict:
    """Forward-return-by-zone (+ coiling streak) over the last `days`, judged HORIZON_DAYS out.
    Computed on the fly from daily bars — read-only, no extra cron."""
    out = {"asOf": datetime.now(timezone.utc).isoformat(), "windowDays": days,
           "n": 0, "horizonDays": HORIZON_DAYS, "byZone": {}, "byStreak": {}, "pending": 0}
    try:
        today = datetime.now(_ET).date()
        since = (today - timedelta(days=days)).isoformat()
        rows = (sb.table(_TABLE).select("session_date,ticker,zone,streak,obs_close")
                .gte("session_date", since).order("session_date").limit(5000).execute().data) or []
        if not rows:
            return out
        from engine.alpaca_client import get_multi_bars
        bars = get_multi_bars(sorted({r["ticker"] for r in rows}), "1Day", days + 25) or {}
        judged, pending = [], 0
        for r in rows:
            df = bars.get(r["ticker"])
            try:
                sdate = datetime.fromisoformat(str(r["session_date"])).date()
            except Exception:
                continue
            fwd = _forward_return(df, sdate, r.get("obs_close")) if df is not None else None
            if fwd is None:
                pending += 1
                continue
            judged.append({**r, "fwd": fwd})
        agg = _aggregate(judged)
        agg.update({"asOf": out["asOf"], "windowDays": days, "pending": pending})
        return agg
    except Exception as e:
        logger.error(f"[churn_history] score failed: {e}")
        return out
