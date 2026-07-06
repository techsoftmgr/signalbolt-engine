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
    rs_x   = bool(bd.get("rs_exempt"))
    # RS-exemption cohort: longs that fired through the regime long-veto because
    # the name had relative strength. Tag the detector label so this experimental
    # cohort shows as its own line ("SMC·RSx") in every detector-keyed view —
    # that's how we watch its realized edge vs the standard detectors before
    # trusting the exemption further.
    if rs_x:
        src = f"{src}·RSx"
    if group_by in ("rs_exempt", "detector_rs_exempt"):
        cohort = "RS-exempt" if rs_x else "Standard"
        return ({"cohort": cohort} if group_by == "rs_exempt"
                else {"detector": src, "cohort": cohort})
    if group_by == "regime":
        return {"regime": regime}
    if group_by == "detector_regime":
        return {"detector": src, "strategy": strat, "regime": regime}
    if group_by == "conviction":
        return {"conviction": _conviction_tier(row.get("confidence_score"))}
    if group_by == "detector_conviction":
        return {"detector": src, "strategy": strat,
                "conviction": _conviction_tier(row.get("confidence_score"))}
    if group_by in ("bucket", "detector_bucket"):
        from engine.regime_buckets import bucket_of
        bkt = bucket_of(regime)
        return {"bucket": bkt} if group_by == "bucket" else {"detector": src, "bucket": bkt}
    if group_by == "detector_direction":
        return {"detector": src, "side": (row.get("direction") or "?").upper()}
    if group_by in ("cmf", "detector_cmf"):
        # Money-flow context at fire time (Chaikin Money Flow state). Answers
        # "do signals fired while money was flowing IN beat those fired during
        # distribution?" — the measure-first gate before wiring CMF into firing.
        cmf_state = bd.get("cmfState") or "unknown"
        return ({"cmf": cmf_state} if group_by == "cmf"
                else {"detector": src, "cmf": cmf_state})
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
    if fields.get("bucket"):
        parts.append(f"[{fields['bucket']}]")
    if fields.get("side"):
        parts.append(f"{fields['side']}")
    if fields.get("conviction"):
        parts.append(f"@{fields['conviction']}")
    if fields.get("cohort"):
        parts.append(str(fields["cohort"]))
    if fields.get("cmf"):
        parts.append(f"flow:{fields['cmf']}")
    return " · ".join(parts) if parts else "—"


def _new_bucket() -> dict:
    return {"n": 0, "wins": 0, "losses": 0, "win_sum": 0.0, "loss_sum": 0.0,
            "pnl_sum": 0.0, "worst": None, "best": None,
            # exit-quality accumulators (from score_breakdown MFE/MAE + timing)
            "mfe_sum": 0.0, "mfe_n": 0, "mae_sum": 0.0, "mae_n": 0,
            "wmae_sum": 0.0, "wmae_n": 0,            # MAE of WINNERS — the stop-tuning stat
            "gb_sum": 0.0, "gb_n": 0,                # give-back: peak MFE − realized
            "tmfe_sum": 0.0, "tmfe_n": 0,            # minutes entry → peak (profit-lock timing)
            "order_n": 0, "mae_first": 0,            # of trades w/ both timings, how many took heat FIRST
            "alpha_sum": 0.0, "alpha_n": 0, "alpha_beat": 0}  # excess vs SPY + how many beat the market


def _accumulate(bucket: dict, pct: float, is_win: bool, sbd: dict | None = None) -> None:
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

    if not isinstance(sbd, dict):
        return
    mfe = sbd.get("mfe_pct")
    if mfe is not None:
        try:
            mfe = float(mfe)
            bucket["mfe_sum"] += mfe; bucket["mfe_n"] += 1
            bucket["gb_sum"] += max(0.0, mfe - pct); bucket["gb_n"] += 1
        except (TypeError, ValueError):
            pass
    mae = sbd.get("mae_pct")
    if mae is not None:
        try:
            mae = float(mae)
            bucket["mae_sum"] += mae; bucket["mae_n"] += 1
            if is_win:
                bucket["wmae_sum"] += mae; bucket["wmae_n"] += 1
        except (TypeError, ValueError):
            pass
    t_mfe = sbd.get("t_mfe_min"); t_mae = sbd.get("t_mae_min")
    if t_mfe is not None:
        try:
            bucket["tmfe_sum"] += float(t_mfe); bucket["tmfe_n"] += 1
        except (TypeError, ValueError):
            pass
    if t_mfe is not None and t_mae is not None:
        try:
            bucket["order_n"] += 1
            if float(t_mae) < float(t_mfe):
                bucket["mae_first"] += 1
        except (TypeError, ValueError):
            bucket["order_n"] -= 1
    alpha = sbd.get("alpha_pct")
    if alpha is not None:
        try:
            alpha = float(alpha)
            bucket["alpha_sum"] += alpha; bucket["alpha_n"] += 1
            if alpha > 0:
                bucket["alpha_beat"] += 1
        except (TypeError, ValueError):
            pass


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

    # Exit-quality block — the tuning lens (separate from the keep/cut verdict):
    #   avg_mfe   how far winners-and-losers ran in our favor at peak
    #   avg_giveback  peak MFE − realized → how much profit we hand back (profit-lock)
    #   winner_mae    avg MAE of WINNERS → how much heat a winner takes; a stop
    #                 tighter than this is cutting winners (min-stop tuning)
    #   avg_mae       avg worst adverse over all trades
    #   avg_t_mfe_min minutes from entry to the favorable peak (when to lock)
    #   mae_before_mfe_pct  % of trades that went AGAINST first then worked —
    #                 high → entries are early/need wider initial stops
    mfe_n, mae_n, wmae_n = bucket["mfe_n"], bucket["mae_n"], bucket["wmae_n"]
    gb_n, tmfe_n, ord_n  = bucket["gb_n"], bucket["tmfe_n"], bucket["order_n"]
    avg_mfe      = round(bucket["mfe_sum"]  / mfe_n,  2) if mfe_n  else None
    avg_mae      = round(bucket["mae_sum"]  / mae_n,  2) if mae_n  else None
    winner_mae   = round(bucket["wmae_sum"] / wmae_n, 2) if wmae_n else None
    avg_giveback = round(bucket["gb_sum"]   / gb_n,   2) if gb_n   else None
    avg_t_mfe_min = round(bucket["tmfe_sum"] / tmfe_n, 1) if tmfe_n else None
    mae_before_mfe_pct = round(100 * bucket["mae_first"] / ord_n, 1) if ord_n else None
    # alpha vs SPY — did this segment make MONEY or just ride the tape?
    alpha_n = bucket["alpha_n"]
    avg_alpha = round(bucket["alpha_sum"] / alpha_n, 2) if alpha_n else None
    market_beat_rate = round(100 * bucket["alpha_beat"] / alpha_n, 1) if alpha_n else None

    return {
        "n": n, "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "payoff": payoff,
        "worst_loss": round(bucket["worst"], 3) if bucket["worst"] is not None else None,
        "best_win":   round(bucket["best"],  3) if bucket["best"]  is not None else None,
        "expectancy_gross": exp_gross, "expectancy_net": exp_net,
        "total_return_pct": total_gross, "net_total_pct": net_total,
        "pnl_per_notional": pnl_cash,
        "verdict": verdict, "reason": reason,
        # exit-quality (None until MFE/MAE + timing accrue)
        "avg_mfe": avg_mfe, "avg_mae": avg_mae, "winner_mae": winner_mae,
        "avg_giveback": avg_giveback, "mfe_sample": mfe_n,
        "avg_t_mfe_min": avg_t_mfe_min, "mae_before_mfe_pct": mae_before_mfe_pct,
        "timing_sample": ord_n,
        # alpha vs SPY (None until enriched): made money vs rode the market
        "avg_alpha": avg_alpha, "market_beat_rate": market_beat_rate, "alpha_sample": alpha_n,
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
                        "conviction", "detector_conviction",
                        "bucket", "detector_bucket", "detector_direction",
                        "rs_exempt", "detector_rs_exempt",
                        "cmf", "detector_cmf"):
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

        sbd = r.get("score_breakdown")
        if not isinstance(sbd, dict):
            sbd = None
        fields = _seg_fields(r, group_by)
        key = _seg_key(fields)
        if key not in segs:
            segs[key] = _new_bucket()
            seg_fields[key] = fields
        _accumulate(segs[key], pct, is_win, sbd)
        _accumulate(portfolio, pct, is_win, sbd)

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
