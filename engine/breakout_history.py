"""
Breakout Watch — track record (aggregate scorecard + per-episode trajectory).

Powers the "Breakout Watch — Track Record" screen. Everything here is computed
ON THE FLY from Alpaca daily bars + the persisted episode rows — no extra table
and no extra cron. Each episode already stores enter_price / entered_at /
trigger_price / triggered_at / exited_at, which is enough to reconstruct:

  • a day-by-day curve of % from the anchor (the TRIGGER if it broke out, else
    the entry — the trigger is the only actionable moment, so we anchor there)
  • MFE (max favorable) and MAE (max adverse) — opportunity vs pain
  • total move (anchor → last/exit close) and days-to-target (+2%)
  • a same-window SPY benchmark, so "edge vs SPY" is visible

The aggregate scorecard rolls these up into the funnel that actually builds
trust: flagged → triggered → followed-through, plus avg MFE/MAE, median
days-to-target, judged accuracy, and the SPY edge.

Best-effort: every failure degrades to a smaller payload, never raises.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("signalbolt.breakout_history")

WIN_PCT = 0.02          # +2% past the anchor = "it paid" (matches breakout_validator)
_TABLE  = "breakout_watch_history"


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


# ── per-episode metrics ──────────────────────────────────────────────────────
def _episode_metrics(ep: dict, bars, spy_bars) -> dict:
    """Daily curve + MFE/MAE/total + days-to-target for one episode.

    bars: that ticker's 1Day DataFrame (UTC index, ohlcv). Anchored to the
    trigger when present (the actionable moment), else to entry.
    """
    out = {"days": [], "mfe_pct": None, "mae_pct": None, "total_pct": None,
           "days_to_target": None, "benchmark_pct": None}
    if bars is None or len(bars) == 0:
        return out

    anchor_ts = _parse(ep.get("triggered_at")) or _parse(ep.get("entered_at"))
    anchor_px = float(ep.get("trigger_price") or ep.get("enter_price") or 0)
    if anchor_ts is None or anchor_px <= 0:
        return out

    end_ts   = _parse(ep.get("exited_at")) or datetime.now(timezone.utc)
    day0     = anchor_ts.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        fwd = bars[(bars.index >= day0) & (bars.index <= end_ts + timedelta(days=1))]
    except Exception:
        return out
    if len(fwd) == 0:
        return out   # episode younger than one daily bar — nothing to chart yet

    days = []
    running_hi = anchor_px
    running_lo = anchor_px
    target_day = None
    for i, (ts, row) in enumerate(fwd.iterrows(), start=1):
        hi = float(row["high"]); lo = float(row["low"]); close = float(row["close"])
        running_hi = max(running_hi, hi)
        running_lo = min(running_lo, lo)
        days.append({
            "day":  i,
            "date": ts.date().isoformat(),
            "close": round(close, 2),
            "pctFromAnchor": round((close - anchor_px) / anchor_px * 100, 2),
        })
        if target_day is None and hi >= anchor_px * (1 + WIN_PCT):
            target_day = i

    last_close = float(fwd["close"].iloc[-1])
    out["days"]            = days
    out["mfe_pct"]         = round((running_hi - anchor_px) / anchor_px * 100, 2)
    out["mae_pct"]         = round((running_lo - anchor_px) / anchor_px * 100, 2)
    out["total_pct"]       = round((last_close - anchor_px) / anchor_px * 100, 2)
    out["days_to_target"]  = target_day

    # SPY move over the same calendar window (the benchmark).
    if spy_bars is not None and len(spy_bars) > 0:
        try:
            sp = spy_bars[(spy_bars.index >= fwd.index[0]) &
                          (spy_bars.index <= fwd.index[-1] + timedelta(days=1))]
            if len(sp) >= 1:
                sp0 = float(sp["close"].iloc[0]); sp1 = float(sp["close"].iloc[-1])
                if sp0 > 0:
                    out["benchmark_pct"] = round((sp1 - sp0) / sp0 * 100, 2)
        except Exception:
            pass
    return out


# ── aggregate scorecard ──────────────────────────────────────────────────────
def _scorecard(eps: list[dict], days: int) -> dict:
    total     = len(eps)
    closed    = [e for e in eps if not e["isOpen"]]
    triggered = [e for e in eps if e.get("triggeredAt")]
    paid      = [e for e in triggered if e.get("daysToTarget")]      # hit +2%
    judged    = [e for e in eps if e.get("outcome") in ("win", "loss")]
    wins      = [e for e in judged if e.get("outcome") == "win"]

    avg_total = _avg([e["totalPct"] for e in eps])
    avg_bench = _avg([e["benchmarkPct"] for e in eps])

    return {
        "windowDays":           days,
        "total":                total,
        "open":                 total - len(closed),
        "closed":               len(closed),
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
def build_history(sb, days: int = 30, limit: int = 80) -> dict:
    """Return {episodes:[...], scorecard:{...}, windowDays} for the track-record screen."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    try:
        eps = (
            sb.table(_TABLE)
              .select("*")
              .gte("session_date", since)
              .order("entered_at", desc=True)
              .limit(max(1, min(limit, 200)))
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
        m = _episode_metrics(ep, bars_by.get(ep.get("ticker")), spy_bars)
        enriched.append({
            "id":            ep.get("id"),
            "ticker":        ep.get("ticker"),
            "state":         ep.get("state"),
            "exitReason":    ep.get("exit_reason"),
            "outcome":       ep.get("outcome"),
            "enteredAt":     ep.get("entered_at"),
            "triggeredAt":   ep.get("triggered_at"),
            "exitedAt":      ep.get("exited_at"),
            "enterPrice":    _num(ep.get("enter_price")),
            "breakoutLevel": _num(ep.get("breakout_level")),
            "triggerPrice":  _num(ep.get("trigger_price")),
            "anchor":        "trigger" if ep.get("triggered_at") else "entry",
            "isOpen":        ep.get("exited_at") is None,
            "mfePct":        m["mfe_pct"],
            "maePct":        m["mae_pct"],
            "totalPct":      m["total_pct"],
            "daysToTarget":  m["days_to_target"],
            "benchmarkPct":  m["benchmark_pct"],
            "realizedPct":   _num(ep.get("realized_pct")),
            "curve":         m["days"],
        })

    return {"episodes": enriched, "scorecard": _scorecard(enriched, days), "windowDays": days}
