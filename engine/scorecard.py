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

# We don't track real position sizes (the engine fires signals, it doesn't size
# trades), so "money made" is expressed on a NORMALIZED notional: equal $ in every
# trade. $1,000/trade by default — total $ = Σ(result_pct/100) × notional.
_NOTIONAL = 1000.0


def _conviction_tier(cs) -> str:
    """Confidence-score band (matches the quant tiers). Lets us see whether a
    detector's HIGH-conviction signals actually beat its low-conviction ones —
    i.e. whether the score is calibrated."""
    try:
        cs = float(cs)
    except (TypeError, ValueError):
        return "?"
    if cs >= 90: return "A+ (90+)"
    if cs >= 80: return "A (80-89)"
    if cs >= 70: return "B+ (70-79)"
    if cs >= 60: return "B (60-69)"
    return "C (<60)"


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
    if group_by == "conviction":
        return {"conviction": _conviction_tier(row.get("confidence_score"))}
    if group_by == "detector_conviction":
        return {"detector": src, "strategy": strat,
                "conviction": _conviction_tier(row.get("confidence_score"))}
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
    if fields.get("conviction"):
        parts.append(f"@{fields['conviction']}")
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


def _stats(bucket: dict, cost_pct: float, min_n: int, notional: float = _NOTIONAL) -> dict:
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

    # "Money made" — position-size-independent: total % return = Σ result_pct
    # (this is n × expectancy, so high win-rate/low payoff vs low win-rate/big
    # payoff is finally comparable in actual profit terms), net of costs, plus a
    # normalized $/notional view.
    total_gross = round(bucket["pnl_sum"], 2)
    net_total   = round(bucket["pnl_sum"] - cost_pct * n, 2) if n else None
    pnl_cash    = round((net_total / 100.0) * notional, 2) if net_total is not None else None

    return {
        "n": n, "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "payoff": payoff,
        "worst_loss": round(bucket["worst"], 3) if bucket["worst"] is not None else None,
        "best_win":   round(bucket["best"],  3) if bucket["best"]  is not None else None,
        "expectancy_gross": exp_gross, "expectancy_net": exp_net,
        "total_return_pct": total_gross, "net_total_pct": net_total,
        "pnl_per_notional": pnl_cash,
        "verdict": verdict, "reason": reason,
    }


def compute(rows: list, group_by: str = "detector",
            cost_pct: float = 0.10, min_n: int = _MIN_N,
            notional: float = _NOTIONAL) -> dict:
    """
    rows: closed-signal dicts with result, result_pct, score_breakdown,
          strategy_type, (optional) regime_type.
    group_by: 'detector' (detector×strategy) | 'regime' | 'detector_regime'.
    notional: $/trade for the normalized cash view (real position sizes aren't
              tracked, so "money made" = Σ result_pct × equal capital per trade).
    Returns {group_by, cost_pct, notional, min_sample, portfolio, segments[]}.
    Each segment also carries profit_share = its net total ÷ portfolio net total
    (what % of all profit this detector contributed).

    A row counts as a WIN when result == 'win' OR (result missing AND pct > 0).
    Rows with result_pct is None are skipped (can't measure edge without P&L).
    """
    if group_by not in ("detector", "regime", "detector_regime",
                        "conviction", "detector_conviction"):
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
        row = {**fields, "label": _label(fields), **_stats(bucket, cost_pct, min_n, notional)}
        out.append(row)

    port = {"label": "ALL SIGNALS", **_stats(portfolio, cost_pct, min_n, notional)}

    # profit_share: each segment's net total as a % of the whole engine's net
    # total — "who's actually carrying the P&L" (only meaningful when the
    # portfolio is net-positive; else left None).
    port_net = port.get("net_total_pct") or 0.0
    for row in out:
        nt = row.get("net_total_pct")
        row["profit_share"] = (round(nt / port_net * 100, 1)
                               if (nt is not None and port_net > 0) else None)

    # Best → worst by net total $ contribution (who made the most money), None last.
    out.sort(key=lambda x: (x["net_total_pct"] is None, -(x["net_total_pct"] or -1e9)))

    return {"group_by": group_by, "cost_pct": cost_pct, "notional": notional,
            "min_sample": min_n, "portfolio": port, "segments": out}
