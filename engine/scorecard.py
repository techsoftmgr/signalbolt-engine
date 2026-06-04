"""
Realized-edge scorecard — PURE aggregation over closed signals.
=================================================================
Turns "win rate" into the honest bottom line: EXPECTANCY per trade, net of
costs, segmented so you can see WHERE the edge is (per detector, per regime, or
both) and how bad the losing tail is.

Why this matters: an 80% win rate is meaningless without payoff. Expectancy =
(win% × avg win) − (loss% × |avg loss|). A detector can win 80% and still bleed
if the 20% losers are large. This module exposes that per segment + a portfolio
roll-up, from already-stored columns (no bar-walking) so it's cheap + testable.

compute() is PURE (given closed-signal rows) → unit-tested. The endpoint in
main.py just fetches rows and calls it.
"""
from __future__ import annotations

# Below this many closed trades a segment's verdict isn't trustworthy.
_MIN_N = 15


def _seg_fields(row: dict, group_by: str) -> dict:
    """Identity fields for a row's segment, per the chosen grouping."""
    bd     = row.get("score_breakdown") or {}
    src    = bd.get("detector_source") or "SMC"
    strat  = row.get("strategy_type") or "—"
    regime = row.get("regime_type") or bd.get("regime_type") or bd.get("regime") or "—"
    if group_by == "regime":
        return {"regime": regime}
    if group_by == "detector_regime":
        return {"detector": src, "strategy": strat, "regime": regime}
    return {"detector": src, "strategy": strat}   # default: detector


def _seg_key(fields: dict) -> tuple:
    return tuple(sorted(fields.items()))


def _label(fields: dict) -> str:
    parts = []
    if fields.get("detector"):
        parts.append(str(fields["detector"]))
    if fields.get("strategy") and fields["strategy"] != "—":
        parts.append(str(fields["strategy"]))
    if fields.get("regime"):
        parts.append(f"[{fields['regime']}]")
    return " · ".join(parts) if parts else "—"


def _new_bucket() -> dict:
    return {"n": 0, "wins": 0, "losses": 0, "win_sum": 0.0, "loss_sum": 0.0,
            "pnl_sum": 0.0, "worst": None, "best": None}


def _accumulate(bucket: dict, pct: float, is_win: bool) -> None:
    bucket["n"] += 1
    bucket["pnl_sum"] += pct
    if is_win:
        bucket["wins"] += 1
        bucket["win_sum"] += pct
    else:
        bucket["losses"] += 1
        bucket["loss_sum"] += pct
    bucket["worst"] = pct if bucket["worst"] is None else min(bucket["worst"], pct)
    bucket["best"]  = pct if bucket["best"]  is None else max(bucket["best"],  pct)


def _stats(bucket: dict, cost_pct: float, min_n: int) -> dict:
    """Derived metrics + verdict from an accumulated bucket."""
    n = bucket["n"]
    win_rate  = round(bucket["wins"] / n * 100, 1) if n else None
    avg_win   = round(bucket["win_sum"]  / bucket["wins"],   3) if bucket["wins"]   else None
    avg_loss  = round(bucket["loss_sum"] / bucket["losses"], 3) if bucket["losses"] else None
    payoff    = round(avg_win / abs(avg_loss), 2) if (avg_win and avg_loss) else None
    exp_gross = round(bucket["pnl_sum"] / n, 3) if n else None
    exp_net   = round(exp_gross - cost_pct, 3) if exp_gross is not None else None

    if n < min_n:
        verdict, reason = "WATCH", f"low sample (n={n})"
    elif exp_net is None:
        verdict, reason = "WATCH", "no data"
    elif exp_net >= 0.10:
        verdict, reason = "KEEP", f"+{exp_net}%/trade after costs"
    elif exp_net <= -0.05:
        verdict, reason = "CUT", f"{exp_net}%/trade after costs"
    else:
        verdict, reason = "WATCH", "marginal edge"

    return {
        "n": n, "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "payoff": payoff,
        "worst_loss": round(bucket["worst"], 3) if bucket["worst"] is not None else None,
        "best_win":   round(bucket["best"],  3) if bucket["best"]  is not None else None,
        "expectancy_gross": exp_gross, "expectancy_net": exp_net,
        "verdict": verdict, "reason": reason,
    }


def compute(rows: list, group_by: str = "detector",
            cost_pct: float = 0.10, min_n: int = _MIN_N) -> dict:
    """
    rows: closed-signal dicts with result, result_pct, score_breakdown,
          strategy_type, (optional) regime_type.
    group_by: 'detector' (detector×strategy) | 'regime' | 'detector_regime'.
    Returns {group_by, cost_pct, min_sample, portfolio, segments[]}.

    A row counts as a WIN when result == 'win' OR (result missing AND pct > 0).
    Rows with result_pct is None are skipped (can't measure edge without P&L).
    """
    if group_by not in ("detector", "regime", "detector_regime"):
        group_by = "detector"

    segs: dict = {}
    seg_fields: dict = {}
    portfolio = _new_bucket()

    for r in rows or []:
        pct = r.get("result_pct")
        if pct is None:
            continue
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            continue
        is_win = (r.get("result") == "win") or (r.get("result") is None and pct > 0)

        fields = _seg_fields(r, group_by)
        key = _seg_key(fields)
        if key not in segs:
            segs[key] = _new_bucket()
            seg_fields[key] = fields
        _accumulate(segs[key], pct, is_win)
        _accumulate(portfolio, pct, is_win)

    out = []
    for key, bucket in segs.items():
        fields = seg_fields[key]
        row = {**fields, "label": _label(fields), **_stats(bucket, cost_pct, min_n)}
        out.append(row)

    # Best → worst by net expectancy (None last).
    out.sort(key=lambda x: (x["expectancy_net"] is None, -(x["expectancy_net"] or -999)))

    port = {"label": "ALL SIGNALS", **_stats(portfolio, cost_pct, min_n)}
    return {"group_by": group_by, "cost_pct": cost_pct, "min_sample": min_n,
            "portfolio": port, "segments": out}
