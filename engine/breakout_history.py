"""
Breakout Watch — track record (aggregate scorecard + per-episode trajectory).

Everything is computed ON THE FLY from Alpaca daily bars + the persisted
episode rows — no extra table, no extra cron.

How an episode is scored (the "real picture"):
  • You enter at the breakout (the trigger). The holding window is the
    HORIZON_DAYS trading days that FOLLOW (a fixed, comparable 1-week window —
    not the trigger day, so day-1 isn't a trivial 0% and pre-breakout intraday
    moves don't count).
  • RESULT = net % at the end of the window (the actual close move). A spike
    that round-trips nets out honestly → it does NOT count as a win.
  • WON = result > 0.  GRADE = magnitude band of the result:
        0–5% C · 5–10% B · 10–15% A · 15%+ A+   (same letters for down moves)
  • MFE (best) / MAE (worst) keep the path's extremes, each with the date it
    occurred.

Best-effort: every failure degrades to a smaller payload, never raises.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("signalbolt.breakout_history")

WIN_PCT      = 0.02     # retained for import compatibility (no longer gates wins)
HORIZON_DAYS = 5        # fixed follow-through window (trading days after the breakout)
_TABLE       = "breakout_watch_history"
_GRADES      = ["A+", "A", "B", "C"]

# Per-bucket semantics. direction = the move that counts as "good" for that
# setup; needsTrigger = whether an episode must break a level (breakout/
# breakdown) — others anchor to entry (no discrete trigger).
_BUCKET_CFG = {
    "breakouts":      {"direction": "up",   "needsTrigger": True,  "label": "Breakout Watch"},
    "breakdowns":     {"direction": "down", "needsTrigger": True,  "label": "Breakdown"},
    "topMomentum":    {"direction": "up",   "needsTrigger": False, "label": "Top Momentum"},
    "pullbacks":      {"direction": "up",   "needsTrigger": False, "label": "Pullback"},
    "highVolumeUp":   {"direction": "up",   "needsTrigger": False, "label": "High Volume ▲ (Accumulation)"},
    "highVolumeDown": {"direction": "down", "needsTrigger": False, "label": "High Volume ▼ (Distribution)"},
    "vwapReclaim":    {"direction": "up",   "needsTrigger": False, "label": "VWAP Reclaim"},
    "oversoldBounce": {"direction": "up",   "needsTrigger": False, "label": "Oversold Bounce"},
}


def bucket_cfg(bucket: str) -> dict:
    return _BUCKET_CFG.get(bucket, _BUCKET_CFG["breakouts"])


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


def _grade(abs_pct: float) -> str:
    if abs_pct >= 15: return "A+"
    if abs_pct >= 10: return "A"
    if abs_pct >= 5:  return "B"
    return "C"


# ── path / result (shared with breakout_validator) ───────────────────────────
def judge_path(bars, anchor_ts, anchor_px, *, horizon_days: int = HORIZON_DAYS,
               direction: str = "up") -> dict:
    """Walk the holding window (days AFTER the breakout); grade by net result.

    direction "up" = a positive net move is a win (bullish buckets); "down" =
    a negative net move is a win (breakdown: the avoid call was right).

    Returns: outcome (win|loss|None-if-open), resultPct (net at window end),
    grade, mfePct/mfeDate (best), maePct/maeDate (worst), daysHeld, and the
    day-by-day curve (% from the breakout).
    """
    out = {"outcome": None, "resultPct": None, "grade": None,
           "mfePct": None, "mfeDate": None, "maePct": None, "maeDate": None,
           "daysHeld": 0, "curve": []}
    if bars is None or len(bars) == 0 or anchor_ts is None or anchor_px <= 0:
        return out

    # Forward window starts the DAY AFTER the breakout (your actual holding period).
    day_after = (anchor_ts + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        fwd = bars[bars.index >= day_after].head(horizon_days)
    except Exception:
        return out
    if len(fwd) == 0:
        return out

    hi_run = anchor_px; lo_run = anchor_px
    hi_date = None; lo_date = None
    curve = []
    for i, (ts, row) in enumerate(fwd.iterrows(), start=1):
        hi = float(row["high"]); lo = float(row["low"]); close = float(row["close"])
        d = ts.date().isoformat()
        if hi > hi_run: hi_run = hi; hi_date = d
        if lo < lo_run: lo_run = lo; lo_date = d
        curve.append({"day": i, "date": d, "close": round(close, 2),
                      "pctFromAnchor": round((close - anchor_px) / anchor_px * 100, 2)})

    last_close = curve[-1]["close"]
    result   = round((last_close - anchor_px) / anchor_px * 100, 2)
    resolved = len(fwd) >= horizon_days
    won = (result < 0) if direction == "down" else (result > 0)
    out.update({
        "outcome":  ("win" if won else "loss") if resolved else None,
        "resultPct": result,
        "grade":     _grade(abs(result)),
        "mfePct":    round((hi_run - anchor_px) / anchor_px * 100, 2),
        "mfeDate":   hi_date or (curve[0]["date"] if curve else None),
        "maePct":    round((lo_run - anchor_px) / anchor_px * 100, 2),
        "maeDate":   lo_date or (curve[0]["date"] if curve else None),
        "daysHeld":  len(curve),
        "curve":     curve,
    })
    return out


# ── per-episode metrics ──────────────────────────────────────────────────────
def _episode_metrics(ep: dict, bars, spy_bars, cfg: dict) -> dict:
    triggered = bool(ep.get("triggered_at"))
    if triggered:
        anchor_ts = _parse(ep.get("triggered_at"))
        anchor_px = float(ep.get("trigger_price") or ep.get("enter_price") or 0)
    else:
        anchor_ts = _parse(ep.get("entered_at"))
        anchor_px = float(ep.get("enter_price") or 0)

    jp = judge_path(bars, anchor_ts, anchor_px, direction=cfg["direction"])

    # For trigger-required buckets (breakout/breakdown), a never-triggered
    # episode means the directional break never came → a loss. Other buckets
    # anchor to entry and are graded purely on the forward move.
    if cfg["needsTrigger"] and not triggered and jp.get("outcome") is not None:
        jp["outcome"] = "loss"

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
def _scorecard(eps: list[dict], days: int, cfg: dict) -> dict:
    total     = len(eps)
    triggered = [e for e in eps if e.get("triggeredAt")]
    judged    = [e for e in eps if e.get("outcome") in ("win", "loss")]
    wins      = [e for e in judged if e.get("outcome") == "win"]
    open_n    = sum(1 for e in eps if e.get("outcome") is None)

    # Grade distribution split by direction (over judged episodes).
    up   = {g: 0 for g in _GRADES}
    down = {g: 0 for g in _GRADES}
    for e in judged:
        g = e.get("grade")
        if g in up:
            (up if (e.get("resultPct") or 0) > 0 else down)[g] += 1

    avg_result = _avg([e["resultPct"] for e in eps])
    avg_bench  = _avg([e["benchmarkPct"] for e in eps])

    return {
        "windowDays":      days,
        "goodDirection":   cfg["direction"],
        "needsTrigger":    cfg["needsTrigger"],
        "total":           total,
        "open":            open_n,
        "closed":          total - open_n,
        "triggered":       len(triggered),
        "triggerRatePct":  round(100 * len(triggered) / total) if total else None,
        "judged":          len(judged),
        "wins":            len(wins),
        "winRatePct":      round(100 * len(wins) / len(judged)) if judged else None,
        "avgResultPct":    avg_result,
        "avgMfePct":       _avg([e["mfePct"] for e in eps]),
        "avgMaePct":       _avg([e["maePct"] for e in eps]),
        "avgBenchmarkPct": avg_bench,
        "edgeVsSpyPct":    round(avg_result - avg_bench, 2)
                           if (avg_result is not None and avg_bench is not None) else None,
        "gradeUp":         up,
        "gradeDown":       down,
    }


# ── public entry point ───────────────────────────────────────────────────────
def build_history(sb, days: int = 30, limit: int = 120, bucket: str = "breakouts") -> dict:
    """Return {episodes, scorecard, windowDays, bucket, label} for one bucket's track record."""
    cfg = bucket_cfg(bucket)
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    try:
        eps = (
            sb.table(_TABLE)
              .select("*")
              .eq("bucket", bucket)
              .gte("session_date", since)
              .order("entered_at", desc=True)
              .limit(max(1, min(limit, 300)))
              .execute()
        ).data or []
    except Exception as e:
        logger.error(f"[breakout_history] fetch failed: {e}")
        return {"episodes": [], "scorecard": _scorecard([], days, cfg), "windowDays": days,
                "bucket": bucket, "label": cfg["label"]}

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
        m = _episode_metrics(ep, bars_by.get(ep.get("ticker")), spy_bars, cfg)
        breakout_at = (ep.get("triggered_at") if triggered else ep.get("entered_at")) or ""
        enriched.append({
            "id":            ep.get("id"),
            "ticker":        ep.get("ticker"),
            "state":         ep.get("state"),
            "exitReason":    ep.get("exit_reason"),
            "outcome":       m["outcome"],
            "won":           (m["outcome"] == "win") if m["outcome"] else None,
            "grade":         m["grade"],
            "enteredAt":     ep.get("entered_at"),
            "triggeredAt":   ep.get("triggered_at"),
            "exitedAt":      ep.get("exited_at"),
            "breakoutAt":    breakout_at,            # full ISO (date + time) of the breakout
            "episodeDate":   breakout_at[:10],       # YYYY-MM-DD for day grouping
            "enterPrice":    _num(ep.get("enter_price")),
            "breakoutLevel": _num(ep.get("breakout_level")),
            "triggerPrice":  _num(ep.get("trigger_price")),
            "anchor":        "trigger" if triggered else "entry",
            "isOpen":        m["outcome"] is None,
            "resultPct":     m["resultPct"],         # net move at window end
            "mfePct":        m["mfePct"],
            "mfeDate":       m["mfeDate"],
            "maePct":        m["maePct"],
            "maeDate":       m["maeDate"],
            "daysHeld":      m["daysHeld"],
            "benchmarkPct":  m["benchmark_pct"],
            "curve":         m["curve"],
        })

    return {"episodes": enriched, "scorecard": _scorecard(enriched, days, cfg),
            "windowDays": days, "bucket": bucket, "label": cfg["label"]}


def build_all_scorecards(sb, days: int = 30) -> dict:
    """Compact scorecard for EVERY bucket (the success cockpit) — one DB query
    + one bars fetch, then a per-bucket scorecard. Used to watch edge-vs-SPY
    accumulate across all sections at a glance."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    try:
        eps = (
            sb.table(_TABLE).select("*")
              .gte("session_date", since)
              .order("entered_at", desc=True)
              .limit(3000)
              .execute()
        ).data or []
    except Exception as e:
        logger.error(f"[breakout_history] all-scorecards fetch failed: {e}")
        eps = []

    tickers  = sorted({e["ticker"] for e in eps if e.get("ticker")})
    bars_by  = {}
    spy_bars = None
    if tickers:
        try:
            from engine.alpaca_client import get_multi_bars
            bars_by  = get_multi_bars(tickers + ["SPY"], "1Day", days + 12) or {}
            spy_bars = bars_by.get("SPY")
        except Exception as e:
            logger.debug(f"[breakout_history] all-scorecards bars failed: {e}")

    by_bucket: dict = {}
    for ep in eps:
        by_bucket.setdefault(ep.get("bucket") or "breakouts", []).append(ep)

    out = []
    for bucket, cfg in _BUCKET_CFG.items():
        rows = []
        for ep in by_bucket.get(bucket, []):
            m = _episode_metrics(ep, bars_by.get(ep.get("ticker")), spy_bars, cfg)
            rows.append({
                "triggeredAt": ep.get("triggered_at"), "outcome": m["outcome"],
                "grade": m["grade"], "resultPct": m["resultPct"],
                "mfePct": m["mfePct"], "maePct": m["maePct"], "benchmarkPct": m["benchmark_pct"],
            })
        out.append({"bucket": bucket, "label": cfg["label"], "scorecard": _scorecard(rows, days, cfg)})

    return {"windowDays": days, "buckets": out}
