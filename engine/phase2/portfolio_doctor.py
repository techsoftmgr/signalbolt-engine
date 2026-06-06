"""
Module #1 — Portfolio Doctor.

Analyzes a trader's holdings → a Portfolio Health Score (0-100) + subscores +
plain-English insights + an AI coach (educational wording, no advice language).
Holdings come from a pluggable broker adapter (engine/phase2/brokers/) — CSV +
Alpaca to start, others stubbed/pluggable.

`analyze()` is PURE on a holdings list + sector map (no I/O) → unit-tested.
`analyze_full()` adds volatility/correlation via a bar fetch. Never raises.
Holding = {ticker, qty, avg_price, current_price?}. Optional cash.
"""
from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger("signalbolt.phase2.portfolio_doctor")


def _sectors():
    try:
        from engine.heatmap_service import TICKER_SECTORS
        return TICKER_SECTORS
    except Exception:
        return {}


def _grade(score):
    return ("A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55
            else "D" if score >= 40 else "F")


def analyze(holdings: list, cash: float = 0.0, sectors: dict | None = None) -> dict:
    """Pure structural health from holdings + cash. No external data."""
    sectors = sectors if sectors is not None else _sectors()
    pos = []
    for h in holdings or []:
        try:
            tk = (h.get("ticker") or "").upper()
            qty = float(h.get("qty") or 0)
            cur = float(h.get("current_price") or h.get("avg_price") or 0)
            avg = float(h.get("avg_price") or cur)
            if not tk or qty <= 0 or cur <= 0:
                continue
            mv = qty * cur
            pos.append({"ticker": tk, "qty": qty, "avg_price": avg, "current_price": cur,
                        "market_value": mv, "unreal_pct": ((cur - avg) / avg * 100) if avg else 0,
                        "sector": sectors.get(tk, "Other")})
        except (TypeError, ValueError):
            continue
    invested = sum(p["market_value"] for p in pos)
    total = invested + max(0.0, float(cash or 0))
    if total <= 0 or not pos:
        return {"score": None, "note": "No positions to analyze."}

    for p in pos:
        p["weight"] = round(p["market_value"] / total * 100, 1)
    cash_pct = round(max(0.0, float(cash or 0)) / total * 100, 1)

    sector_w = defaultdict(float)
    for p in pos:
        sector_w[p["sector"]] += p["weight"]
    max_pos = max(p["weight"] for p in pos)
    max_sector = max(sector_w.values()) if sector_w else 0
    hhi = sum((p["weight"] / 100) ** 2 for p in pos)        # Herfindahl concentration

    # ── subscores (0-100, higher = healthier) ──
    sub = {
        "position_concentration": int(round(max(0, 100 - max(0, max_pos - 15) * 4))),
        "sector_concentration":   int(round(max(0, 100 - max(0, max_sector - 30) * 2.5))),
        "diversification":        int(round(max(0, min(100, 100 - (hhi - 1 / max(1, len(pos))) * 250)))),
        "cash_management":        int(round(100 - abs(cash_pct - 10) * 2.5)) if cash_pct <= 50 else 40,
        "correlation_exposure":   int(round(max(0, 100 - max(0, max_sector - 25) * 2))),
    }
    sub = {k: max(0, min(100, v)) for k, v in sub.items()}
    score = int(round(sum(sub.values()) / len(sub)))

    strengths, risks = [], []
    if cash_pct >= 5:
        strengths.append(f"Healthy cash position ({cash_pct}%)")
    if max_sector <= 35:
        strengths.append("Reasonably balanced sector allocation")
    if max_pos > 20:
        risks.append(f"Single position exceeds 20% ({max(pos, key=lambda p: p['weight'])['ticker']} at {max_pos}%)")
    if max_sector > 40:
        top_sec = max(sector_w, key=sector_w.get)
        risks.append(f"{int(max_sector)}% concentrated in {top_sec}")
    if len(pos) < 5:
        risks.append(f"Only {len(pos)} positions — limited diversification")
    if cash_pct == 0:
        risks.append("No cash buffer")

    winners = sorted(pos, key=lambda p: -p["unreal_pct"])
    return {
        "score": score, "grade": _grade(score), "subscores": sub,
        "positions": sorted(pos, key=lambda p: -p["weight"]),
        "invested": round(invested, 2), "cash": round(float(cash or 0), 2),
        "cash_pct": cash_pct, "total_value": round(total, 2),
        "sector_allocation": {k: round(v, 1) for k, v in sorted(sector_w.items(), key=lambda x: -x[1])},
        "largest_position": sorted(pos, key=lambda p: -p["weight"])[0],
        "largest_winner": winners[0] if winners else None,
        "largest_loser": winners[-1] if winners else None,
        "strengths": strengths, "risks": risks,
    }


def coach(report: dict) -> str:
    """Plain-English educational summary. No advice language."""
    if not report or report.get("score") is None:
        return "Connect a brokerage or upload a CSV to analyze your portfolio."
    parts = [f"Your portfolio health score is {report['score']}/100 (grade {report.get('grade')})."]
    risks = report.get("risks") or []
    if risks:
        parts.append("Areas to be aware of: " + "; ".join(risks[:3]) + ".")
        sa = report.get("sector_allocation") or {}
        if sa:
            top, w = next(iter(sa.items()))
            if w >= 40:
                parts.append(f"A correction in {top} could impact roughly {int(w)}% of your holdings — "
                             f"concentration risk is worth understanding.")
    else:
        parts.append("No major structural concentration risks stand out.")
    parts.append("This is educational analysis, not financial advice.")
    return " ".join(parts)


def analyze_full(holdings: list, cash: float = 0.0) -> dict:
    """analyze() + volatility/beta enrichment via a bar fetch. Never raises."""
    rep = analyze(holdings, cash)
    if rep.get("score") is None:
        rep["coach"] = coach(rep)
        rep["enabled"] = True
        return rep
    try:
        import numpy as np
        from engine.alpaca_client import get_multi_bars
        tks = [p["ticker"] for p in rep["positions"]]
        bars = get_multi_bars(tks + ["SPY"], "1Day", 90) or {}
        spy = bars.get("SPY")
        spy_ret = np.diff(spy["close"].values) / spy["close"].values[:-1] if spy is not None and len(spy) > 30 else None
        betas, vols = {}, {}
        for tk in tks:
            df = bars.get(tk)
            if df is None or len(df) < 30:
                continue
            r = np.diff(df["close"].values) / df["close"].values[:-1]
            vols[tk] = float(np.std(r[-60:]) * np.sqrt(252) * 100)
            if spy_ret is not None and len(r) == len(spy_ret):
                cov = np.cov(r[-60:], spy_ret[-60:])
                betas[tk] = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] else 1.0
        # weighted portfolio beta + vol
        wsum = sum(p["weight"] for p in rep["positions"])
        pbeta = sum(p["weight"] * betas.get(p["ticker"], 1.0) for p in rep["positions"]) / wsum if wsum else 1.0
        pvol = sum(p["weight"] * vols.get(p["ticker"], 25.0) for p in rep["positions"]) / wsum if wsum else 25.0
        rep["portfolio_beta"] = round(pbeta, 2)
        rep["portfolio_volatility_pct"] = round(pvol, 1)
        rep["subscores"]["volatility_exposure"] = int(max(0, min(100, 100 - max(0, pvol - 25) * 2)))
        rep["score"] = int(round(sum(rep["subscores"].values()) / len(rep["subscores"])))
        rep["grade"] = _grade(rep["score"])
    except Exception as e:
        logger.debug(f"[portfolio_doctor] enrich failed: {e}")
    rep["coach"] = coach(rep)
    rep["enabled"] = True
    return rep
