"""
Breakout Watch — track record (aggregate scorecard + per-episode trajectory).

Everything is computed ON THE FLY from Alpaca daily bars + the persisted
episode rows — no extra table, no extra cron.

Resolution is PATH-DEPENDENT and BOUNDED (the "intelligent day count"):
each triggered breakout is walked forward from the trigger and resolves on the
FIRST of —
  • TARGET  — a daily high reaches trigger × (1 + WIN_PCT)  → win
  • STOP    — a daily close falls back below the breakout level → loss
  • HORIZON — HORIZON_DAYS elapse with neither → loss (no follow-through)
The day-by-day curve therefore stops itself when the trade is decided; it is
never an arbitrary fixed length. A win realises at the target (what you'd
actually capture), not at a transient spike that later round-trips.

Episodes that never triggered (FADED / EXPIRED) are losses by definition —
the watch flagged a breakout that never came.

Best-effort: every failure degrades to a smaller payload, never raises.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("signalbolt.breakout_history")

WIN_PCT      = 0.02     # +2% past the trigger = target hit
HORIZON_DAYS = 5        # a breakout gets this many trading days to follow through
_TABLE       = "breakout_watch_history"


# ── small helpers ───────────────────────────────────────────────────────────
def _parse(ts):
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _num(v):
    try:
        return round(float(v), 2) if v is not None else None
    except Exception:
        return None


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _median(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    n = len(vals); mid = n // 2
    return vals[mid] if n % 2 else round((vals[mid - 1] + vals[mid]) / 2, 1)


# ── path-dependent resolution (shared with breakout_validator) ───────────────
def judge_path(bars, anchor_ts, anchor_px, stop_level,
               *, horizon_days: int = HORIZON_DAYS, win_pct: float = WIN_PCT) -> dict:
    """Walk forward from the anchor; resolve on target-before-stop within horizon.

    Returns: outcome (win|loss|None-if-open), resolutionType (target|stop|
    horizon|open), daysToResolve, resolvedDate, realizedPct, mfePct, maePct,
    and the bounded day-by-day curve (anchor → resolution).
    """
    out = {"outcome": None, "resolutionType": None, "daysToResolve": None,
           "resolvedDate": None, "realizedPct": None, "mfePct": None,
           "maePct": None, "curve": []}
    if bars is None or len(bars) == 0 or anchor_ts is None or anchor_px <= 0:
        return out

    # Forward window starts the DAY AFTER the breakout. You enter at the trigger
    # (its close), so the holding period — and MFE/MAE — is the days that follow.
    # Including the trigger day would make day-1 always 0% (close vs itself) and
    # would count the trigger day's pre-breakout intraday low as drawdown you
    # never actually sat through.
    day_after = (anchor_ts + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        fwd = bars[bars.index >= day_after].head(horizon_days)
    except Exception:
        return out
    if len(fwd) == 0:
        return out

    target = anchor_px * (1 + win_pct)
    stop   = float(stop_level) if stop_level else None
    hi_run = anchor_px; lo_run = anchor_px
    curve  = []
    rtype  = None; resolved_i = None
    for i, (ts, row) in enumerate(fwd.iterrows(), start=1):
        hi = float(row["high"]); lo = float(row["low"]); close = float(row["close"])
        hi_run = max(hi_run, hi); lo_run = min(lo_run, lo)
        curve.append({"day": i, "date": ts.date().isoformat(), "close": round(close, 2),
                      "pctFromAnchor": round((close - anchor_px) / anchor_px * 100, 2)})
        if hi >= target:                       # target reached intraday → win
            rtype = "target"; resolved_i = i; break
        if stop is not None and close < stop:  # daily close back below the level → fail
            rtype = "stop"; resolved_i = i; break

    last_close = curve[-1]["close"]
    if rtype == "target":
        outcome, realized = "win", round(win_pct * 100, 2)
    elif rtype == "stop":
        outcome, realized = "loss", round((last_close - anchor_px) / anchor_px * 100, 2)
    elif len(fwd) >= horizon_days:             # full horizon, never resolved → no follow-through
        rtype, outcome = "horizon", "loss"
        realized = round((last_close - anchor_px) / anchor_px * 100, 2)
    else:                                      # not enough forward bars yet → still open
        rtype, outcome = "open", None
        realized = round((last_close - anchor_px) / anchor_px * 100, 2)

    out.update({
        "outcome": outcome, "resolutionType": rtype,
        "daysToResolve": resolved_i, "resolvedDate": curve[-1]["date"],
        "realizedPct": realized,
        "mfePct": round((hi_run - anchor_px) / anchor_px * 100, 2),
        "maePct": round((lo_run - anchor_px) / anchor_px * 100, 2),
        "curve": curve,
    })
    return out


# ── per-episode metrics ──────────────────────────────────────────────────────
def _episode_metrics(ep: dict, bars, spy_bars) -> dict:
    triggered = bool(ep.get("triggered_at"))
    if triggered:
        anchor_ts  = _parse(ep.get("triggered_at"))
        anchor_px  = float(ep.get("trigger_price") or ep.get("enter_price") or 0)
        stop_level = ep.get("breakout_level")
    else:
        anchor_ts  = _parse(ep.get("entered_at"))
        anchor_px  = float(ep.get("enter_price") or 0)
        stop_level = None

    jp = judge_path(bars, anchor_ts, anchor_px, stop_level)

    # Never triggered → the watch's breakout call didn't come: it's a loss.
    if not triggered and jp.get("outcome") is not None:
        jp["outcome"]        = "loss"
        jp["resolutionType"] = (ep.get("exit_reason") or "FADED").lower()
        jp["daysToResolve"]  = None

    # SPY over the same window (benchmark).
    bench = None
    curve = jp.get("curve") or []
    if spy_bars is not None and len(spy_bars) and curve:
        try:
            lo = _parse(curve[0]["date"]); hi = _parse(curve[-1]["date"])
            sp = spy_bars[(spy_bars.index >= lo) & (spy_bars.index <= hi + timedelta(days=1))]
            if len(sp) >= 1:
                sp0 = float(sp["close"].iloc[0]); sp1 = float(sp["close"].iloc[-1])
                if sp0 > 0:
                    bench = round((sp1 - sp0) / sp0 * 100, 2)
        except Exception:
            pass
    jp["benchmark_pct"] = bench
    return jp


# ── aggregate scorecard ──────────────────────────────────────────────────────
def _scorecard(eps: list[dict], days: int) -> dict:
    total     = len(eps)
    triggered = [e for e in eps if e.get("triggeredAt")]
    paid      = [e for e in triggered if e.get("resolution") == "target"]
    judged    = [e for e in eps if e.get("outcome") in ("win", "loss")]
    wins      = [e for e in judged if e.get("outcome") == "win"]
    open_n    = sum(1 for e in eps if e.get("outcome") is None)

    avg_total = _avg([e["totalPct"] for e in eps])
    avg_bench = _avg([e["benchmarkPct"] for e in eps])

    return {
        "windowDays":           days,
        "total":                total,
        "open":                 open_n,
        "closed":               total - open_n,
        "triggered":            len(triggered),
        "triggerRatePct":       round(100 * len(triggered) / total) if total else None,
        "followThrough":        len(paid),
        "followThroughRatePct": round(100 * len(paid) / len(triggered)) if triggered else None,
        "judged":               len(judged),
        "wins":                 len(wins),
        "accuracyPct":          round(100 * len(wins) / len(judged)) if judged else None,
        "avgMfePct":            _avg([e["mfePct"] for e in eps]),
        "avgMaePct":            _avg([e["maePct"] for e in eps]),
        "avgTotalPct":          avg_total,
        "avgBenchmarkPct":      avg_bench,
        "edgeVsSpyPct":         round(avg_total - avg_bench, 2)
                                if (avg_total is not None and avg_bench is not None) else None,
        "medianDaysToTarget":   _median([e["daysToTarget"] for e in paid]),
    }


# ── public entry point ───────────────────────────────────────────────────────
def build_history(sb, days: int = 30, limit: int = 120) -> dict:
    """Return {episodes:[...], scorecard:{...}, windowDays} for the track-record screen."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    try:
        eps = (
            sb.table(_TABLE)
              .select("*")
              .gte("session_date", since)
              .order("entered_at", desc=True)
              .limit(max(1, min(limit, 300)))
              .execute()
        ).data or []
    except Exception as e:
        logger.error(f"[breakout_history] fetch failed: {e}")
        return {"episodes": [], "scorecard": _scorecard([], days), "windowDays": days}

    tickers  = sorted({e["ticker"] for e in eps if e.get("ticker")})
    bars_by  = {}
    spy_bars = None
    if tickers:
        try:
            from engine.alpaca_client import get_multi_bars
            bars_by  = get_multi_bars(tickers + ["SPY"], "1Day", days + 12) or {}
            spy_bars = bars_by.get("SPY")
        except Exception as e:
            logger.debug(f"[breakout_history] bars fetch failed: {e}")

    enriched = []
    for ep in eps:
        triggered = bool(ep.get("triggered_at"))
        m = _episode_metrics(ep, bars_by.get(ep.get("ticker")), spy_bars)
        ep_date = (ep.get("triggered_at") if triggered else ep.get("entered_at")) or ""
        enriched.append({
            "id":            ep.get("id"),
            "ticker":        ep.get("ticker"),
            "state":         ep.get("state"),
            "exitReason":    ep.get("exit_reason"),
            "outcome":       m["outcome"],                       # computed live (path-dependent)
            "resolution":    m["resolutionType"],                # target|stop|horizon|faded|expired|open
            "enteredAt":     ep.get("entered_at"),
            "triggeredAt":   ep.get("triggered_at"),
            "exitedAt":      ep.get("exited_at"),
            "episodeDate":   ep_date[:10],                       # YYYY-MM-DD for day grouping
            "enterPrice":    _num(ep.get("enter_price")),
            "breakoutLevel": _num(ep.get("breakout_level")),
            "triggerPrice":  _num(ep.get("trigger_price")),
            "anchor":        "trigger" if triggered else "entry",
            "isOpen":        m["outcome"] is None,
            "mfePct":        m["mfePct"],
            "maePct":        m["maePct"],
            "totalPct":      m["realizedPct"],                   # realised at resolution
            "daysToTarget":  m["daysToResolve"] if m["resolutionType"] == "target" else None,
            "daysToResolve": m["daysToResolve"],
            "resolvedDate":  m["resolvedDate"],
            "benchmarkPct":  m["benchmark_pct"],
            "curve":         m["curve"],
        })

    return {"episodes": enriched, "scorecard": _scorecard(enriched, days), "windowDays": days}
