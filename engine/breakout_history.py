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
VOL_CONFIRM  = 1.5      # break-day volume ≥ 1.5× the prior-20d avg = volume-confirmed
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
    "turnaround":     {"direction": "up",   "needsTrigger": False, "label": "Turnaround",          "horizonDays": 50, "winMfePct": 8},
    "peak":           {"direction": "down", "needsTrigger": False, "label": "Peak / Distribution", "horizonDays": 50, "winMfePct": 8},
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


def _vol_ratio(bars, anchor_ts):
    """Break-day volume ÷ the prior-20-day avg volume (None if unknown)."""
    if bars is None or len(bars) < 21 or anchor_ts is None:
        return None
    try:
        cutoff = anchor_ts.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        prior = bars[bars.index <= cutoff]      # bars up to & incl. the break day
        if len(prior) < 21:
            return None
        vols = prior["volume"].values.astype(float)
        avg20 = float(vols[-21:-1].mean())
        return round(float(vols[-1]) / avg20, 2) if avg20 > 0 else None
    except Exception:
        return None


# ── path / result (shared with breakout_validator) ───────────────────────────
def judge_path(bars, anchor_ts, anchor_px, *, horizon_days: int = HORIZON_DAYS,
               direction: str = "up", win_mfe_pct=None) -> dict:
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

    # ── Triple-barrier labelling (when win_mfe_pct is set, i.e. cycle buckets) ──
    # Resolve at the FIRST of:
    #   • target  (+win% favourable)             → WIN  (a real swing fired)
    #   • stop    (STRUCTURAL: base/cap. break)  → LOSS (thesis invalidated)
    #   • horizon (vertical barrier)             → judge on the net move (chop)
    # Clear winners and knives resolve in days/weeks; only sideways basers wait
    # out the backstop, so the exact horizon length barely matters.
    tgt_px = stp_px = None
    if win_mfe_pct is not None:
        # STRUCTURAL stop: a DECISIVE break of the base/capitulation extreme over
        # the ~20 bars before entry — gives a confirmed swing room to breathe
        # (HOOD can dip within its base) instead of a flat −8% that normal chop
        # would trip. Floored so the stop is never tighter than 2% from entry.
        _SLB = 20; _SBUF = 0.015
        try:
            _pre = bars[bars.index <= anchor_ts].tail(_SLB)
        except Exception:
            _pre = None
        if direction == "down":   # peak: win = price FALLS; stop = squeezed UP through the base high
            tgt_px = anchor_px * (1 - win_mfe_pct / 100)
            _base_hi = float(_pre["high"].max()) if (_pre is not None and len(_pre) >= 5) else anchor_px * (1 + win_mfe_pct / 100)
            stp_px = max(_base_hi * (1 + _SBUF), anchor_px * 1.02)
        else:                     # turnaround: win = price RISES; stop = break BELOW the base low
            tgt_px = anchor_px * (1 + win_mfe_pct / 100)
            _base_lo = float(_pre["low"].min()) if (_pre is not None and len(_pre) >= 5) else anchor_px * (1 - win_mfe_pct / 100)
            stp_px = min(_base_lo * (1 - _SBUF), anchor_px * 0.98)
    barrier = None; barrier_day = None; barrier_close = None

    for i, (ts, row) in enumerate(fwd.iterrows(), start=1):
        hi = float(row["high"]); lo = float(row["low"]); close = float(row["close"])
        d = ts.date().isoformat()
        if hi > hi_run: hi_run = hi; hi_date = d
        if lo < lo_run: lo_run = lo; lo_date = d
        curve.append({"day": i, "date": d, "close": round(close, 2),
                      "pctFromAnchor": round((close - anchor_px) / anchor_px * 100, 2)})
        if tgt_px is not None and barrier is None:
            hit_tgt = (lo <= tgt_px) if direction == "down" else (hi >= tgt_px)
            hit_stp = (hi >= stp_px) if direction == "down" else (lo <= stp_px)
            if hit_tgt:
                barrier, barrier_day, barrier_close = "win", i, close
            elif hit_stp:
                barrier, barrier_day, barrier_close = "loss", i, close
            if barrier:
                break   # exit at the first barrier touched

    result = round(((barrier_close if barrier_close is not None else curve[-1]["close"]) - anchor_px) / anchor_px * 100, 2)
    mfe    = round((hi_run - anchor_px) / anchor_px * 100, 2)
    mae    = round((lo_run - anchor_px) / anchor_px * 100, 2)

    if win_mfe_pct is not None:
        if barrier is not None:                       # target / stop hit → resolved early
            outcome    = barrier
            days_held  = barrier_day
            grade_move = win_mfe_pct
        elif len(fwd) >= horizon_days:                # vertical barrier → judge net move
            won        = (result < 0) if direction == "down" else (result > 0)
            outcome    = "win" if won else "loss"
            days_held  = len(curve)
            grade_move = abs(result)
        else:                                         # not enough forward data yet → open
            outcome    = None
            days_held  = len(curve)
            grade_move = abs(result)
    else:                                             # non-cycle buckets: net-at-horizon
        resolved   = len(fwd) >= horizon_days
        won        = (result < 0) if direction == "down" else (result > 0)
        outcome    = ("win" if won else "loss") if resolved else None
        days_held  = len(curve)
        grade_move = abs(result)

    out.update({
        "outcome":   outcome,
        "resultPct": result,
        "grade":     _grade(grade_move),
        "stopPct":   (round((stp_px / anchor_px - 1) * 100, 2) if stp_px else None),
        "mfePct":    mfe,
        "mfeDate":   hi_date or (curve[0]["date"] if curve else None),
        "maePct":    mae,
        "maeDate":   lo_date or (curve[0]["date"] if curve else None),
        "daysHeld":  days_held,
        "curve":     curve,
    })

    # Prepend the ENTRY day (anchor) as the curve's day-0 origin, so the track
    # record visibly STARTS on the date you got in (e.g. June 1) at its 0%
    # baseline. The forward curve deliberately begins the day AFTER entry (the
    # real holding window) for outcome math — but hiding the entry day made it
    # look like June 1 had "no %". This is display-only: outcome / resultPct /
    # mfe / mae / daysHeld above are all computed from the forward window first.
    try:
        # Day-0 = the entry/trigger day itself, shown with its ACTUAL % (how it
        # closed that day vs the anchor price) — not a flat 0%, which read as
        # "no data" for the first day (e.g. June 1). For an intraday trigger this
        # is the rest-of-day move after the break.
        _ad = bars[bars.index <= anchor_ts]
        _entry_close = float(_ad["close"].iloc[-1]) if len(_ad) else anchor_px
        out["curve"] = [{
            "day": 0,
            "date": anchor_ts.date().isoformat(),
            "close": round(_entry_close, 2),
            "pctFromAnchor": round((_entry_close - anchor_px) / anchor_px * 100, 2),
        }] + out["curve"]
    except Exception:
        pass
    return out


def forming_curve(bars, forming_ts, forming_px, *, days: int = HORIZON_DAYS,
                  direction: str = "up") -> dict:
    """Daily % path anchored at the FORMING / flag point, INCLUSIVE of the flag
    day (day 1). Answers "from the moment we flagged it forming, how did it move
    each day?" — measures whether the early flag itself was predictive, alongside
    the trigger-anchored follow-through curve. Returns {curve, resultPct, daysHeld}.
    """
    out = {"curve": [], "resultPct": None, "daysHeld": 0}
    if bars is None or len(bars) == 0 or forming_ts is None or not forming_px or forming_px <= 0:
        return out
    start = forming_ts.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        fwd = bars[bars.index >= start].head(days)
    except Exception:
        return out
    if len(fwd) == 0:
        return out
    curve = []
    for i, (ts, row) in enumerate(fwd.iterrows(), start=1):
        close = float(row["close"])
        curve.append({
            "day": i, "date": ts.date().isoformat(), "close": round(close, 2),
            "pctFromForming": round((close - forming_px) / forming_px * 100, 2),
        })
    out["curve"]     = curve
    out["resultPct"] = curve[-1]["pctFromForming"] if curve else None
    out["daysHeld"]  = len(curve)
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

    jp = judge_path(bars, anchor_ts, anchor_px, direction=cfg["direction"],
                    horizon_days=int(cfg.get("horizonDays", HORIZON_DAYS)),
                    win_mfe_pct=cfg.get("winMfePct"))

    # For trigger-required buckets (breakout/breakdown), a never-triggered
    # episode means the directional break never came → a loss. Other buckets
    # anchor to entry and are graded purely on the forward move.
    if cfg["needsTrigger"] and not triggered and jp.get("outcome") is not None:
        jp["outcome"] = "loss"

    # Volume confirmation at the break day (for strong-vs-weak segmentation).
    vr = _vol_ratio(bars, anchor_ts)
    jp["volRatio"]     = vr
    jp["volConfirmed"] = (vr >= VOL_CONFIRM) if vr is not None else None

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

    # Forming-anchored path: from the WATCH/flag point (entered_at / enter_price),
    # inclusive of the flag day. Separate from the trigger-anchored follow-through
    # above — for breakouts the flag precedes the break, so this shows the run
    # FROM when we first flagged it forming.
    try:
        f_ts = _parse(ep.get("entered_at"))
        f_px = float(ep.get("enter_price") or 0)
        fc = forming_curve(bars, f_ts, f_px,
                           days=int(cfg.get("horizonDays", HORIZON_DAYS)) + 5,
                           direction=cfg["direction"])
        jp["formingCurve"]     = fc["curve"]
        jp["formingResultPct"] = fc["resultPct"]
        jp["formingDays"]      = fc["daysHeld"]
    except Exception:
        jp["formingCurve"] = []
        jp["formingResultPct"] = None
        jp["formingDays"] = 0
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

    # Strong-vs-weak-volume segmentation — does the volume confirmation pay?
    def _seg(subset):
        judged_s = [e for e in subset if e.get("outcome") in ("win", "loss")]
        wins_s   = [e for e in judged_s if e.get("outcome") == "win"]
        ar = _avg([e["resultPct"] for e in subset])
        ab = _avg([e["benchmarkPct"] for e in subset])
        return {
            "n": len(subset), "judged": len(judged_s), "wins": len(wins_s),
            "winRatePct":   round(100 * len(wins_s) / len(judged_s)) if judged_s else None,
            "avgResultPct": ar,
            "edgeVsSpyPct": round(ar - ab, 2) if (ar is not None and ab is not None) else None,
        }

    return {
        "windowDays":      days,
        "goodDirection":   cfg["direction"],
        "volStrong":       _seg([e for e in eps if e.get("volConfirmed") is True]),
        "volWeak":         _seg([e for e in eps if e.get("volConfirmed") is False]),
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
            "volRatio":      m.get("volRatio"),
            "volConfirmed":  m.get("volConfirmed"),
            "curve":         m["curve"],
            "formingCurve":     m.get("formingCurve") or [],   # % from the forming/flag day (incl. day 1)
            "formingResultPct": m.get("formingResultPct"),
            "formingDays":      m.get("formingDays") or 0,
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
                "volConfirmed": m.get("volConfirmed"),
            })
        out.append({"bucket": bucket, "label": cfg["label"], "scorecard": _scorecard(rows, days, cfg)})

    return {"windowDays": days, "buckets": out}
