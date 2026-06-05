"""
Alpha vs the market — did a detector actually make MONEY, or just ride the tape?

For each closed signal we record the return of the equivalent SPY exposure over
the same hold and the excess (alpha). Direction-aware: a LONG's benchmark is
SPY's return over the hold; a SHORT's benchmark is −SPY (being short the
market). alpha = realized result − benchmark.

Stored in score_breakdown.{benchmark_return_pct, alpha_pct} (JSONB, no
migration). The scorecard then aggregates avg_alpha + market-beat-rate per
detector / regime, so a +EV detector in a +20% tape (rode beta) is finally
distinguishable from one that beat a flat/down tape (real alpha).

Benchmark convention: SPY OPEN on the entry day → SPY CLOSE on the exit day
(daily bars). Same-day trades get that day's open→close; multi-day get
entry-open→exit-close. A pragmatic excess-vs-market read, not strict CAPM
(beta is assumed 1) — enough to answer "beat the market or not".

`position_alpha()` is PURE (unit-tested). `enrich()` / `backfill()` do I/O,
best-effort, never raise.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("signalbolt.alpha")


def position_alpha(direction, result_pct, spy_entry, spy_exit) -> dict | None:
    """Pure: {benchmark_return_pct, alpha_pct} for a closed position, or None."""
    try:
        e = float(spy_entry); x = float(spy_exit); r = float(result_pct)
    except (TypeError, ValueError):
        return None
    if e <= 0:
        return None
    spy_ret = (x - e) / e * 100.0
    is_long = (direction or "").upper() == "LONG"
    bench = spy_ret if is_long else -spy_ret      # the market exposure the position carries
    return {"benchmark_return_pct": round(bench, 2), "alpha_pct": round(r - bench, 2)}


def _spy_oc_by_date(days: int) -> dict:
    """{date: (open, close)} of SPY daily bars over the window. {} on failure."""
    try:
        from engine.alpaca_client import get_bars
        df = get_bars("SPY", "1Day", days + 6)
        if df is None or df.empty:
            return {}
        out = {}
        for idx, row in df.iterrows():
            try:
                d = idx.date() if hasattr(idx, "date") else idx
                out[d] = (float(row["open"]), float(row["close"]))
            except Exception:
                pass
        return out
    except Exception as e:
        logger.debug(f"[alpha] SPY bar fetch failed: {e}")
        return {}


def enrich(sb, signals: list, spy_oc: dict) -> int:
    """Write benchmark_return_pct + alpha_pct onto closed signals missing them.
    Returns the count enriched. Best-effort."""
    if not signals or not spy_oc:
        return 0
    dates = sorted(spy_oc)

    def _open_on(d):
        if d in spy_oc:
            return spy_oc[d][0]
        prior = [x for x in dates if x <= d]
        return spy_oc[prior[-1]][0] if prior else None

    def _close_on(d):
        if d in spy_oc:
            return spy_oc[d][1]
        prior = [x for x in dates if x <= d]
        return spy_oc[prior[-1]][1] if prior else None

    n = 0
    for s in signals:
        sbd = s.get("score_breakdown")
        if not isinstance(sbd, dict) or sbd.get("alpha_pct") is not None:
            continue
        if s.get("result_pct") is None or not s.get("created_at") or not s.get("closed_at"):
            continue
        try:
            ed = datetime.fromisoformat(s["created_at"].replace("Z", "+00:00")).date()
            xd = datetime.fromisoformat(s["closed_at"].replace("Z", "+00:00")).date()
        except Exception:
            continue
        a = position_alpha(s.get("direction"), s["result_pct"], _open_on(ed), _close_on(xd))
        if not a:
            continue
        merged = dict(sbd); merged.update(a)
        try:
            sb.table("signals").update({"score_breakdown": merged}).eq("id", s["id"]).execute()
            s["score_breakdown"] = merged
            n += 1
        except Exception as e:
            logger.debug(f"[alpha] update failed for {s.get('id')}: {e}")
    return n


def backfill(sb, days: int = 45) -> int:
    """One-shot / periodic: enrich every closed signal in the window lacking
    alpha. Returns count enriched. Never raises."""
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = (sb.table("signals")
                .select("id,direction,result_pct,created_at,closed_at,score_breakdown")
                .eq("status", "closed").gte("closed_at", since).limit(5000).execute().data) or []
        spy = _spy_oc_by_date(days + 10)
        n = enrich(sb, rows, spy)
        logger.info(f"[alpha] backfill enriched {n}/{len(rows)} closed signals ({days}d)")
        return n
    except Exception as e:
        logger.error(f"[alpha] backfill failed: {e}")
        return 0
