"""
Trend-ride scorecard — does letting confirmed-green swings RUN actually pay?

Measures the trend_ride feature (engine/trend_ride.py, PR #362) against the early-exit
behaviour it replaced. Reads CLOSED swing signals and segments them on the durable
`score_breakdown.trend_ride_ever` marker (set-once when a trade entered ride mode):

  • RODE      — entered trend-ride at some point
  • DID NOT   — swing-eligible but never qualified (the comparison group)

Headline questions it answers:
  1. Do RODE swings out-expectancy the swings that DIDN'T ride?
  2. Of rides that ended on `trend_break` (decisive daily close back through the 20-MA),
     do they beat the old early exits (structure_reversal / market_close, which the
     post-mortem clocked at ~ -1.2% / -0.6%)?
  3. The failure mode to watch: "rode and GAVE IT BACK" — a ride whose peak (mfe_pct)
     was big but realized result ended ≤0. If this count is high the gate is too loose.
  4. Did `structure_reversal` exits on swings actually DROP after the feature (they
     should — riding suppresses them)?

Pure `summarize(rows)` (unit-tested) + `build(sb, days)` (fetch + summarize) + a CLI.
Read-only — never writes. Run:  python -m engine.trend_ride_scorecard --days 30
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from engine import trend_ride

# Early-exit reasons the feature is meant to REPLACE on a working swing.
_EARLY_EXITS = ("structure_reversal", "market_close")
# "Gave it back" thresholds: peak gain was real, realized ended flat/negative.
_GAVEBACK_MFE_MIN = 3.0
_GAVEBACK_RESULT_MAX = 0.5


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _stats(group: list[dict]) -> dict:
    n = len(group)
    if not n:
        return {"n": 0, "win_pct": 0.0, "exp": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "pf": 0.0, "total": 0.0, "avg_mfe": 0.0}
    pcts = [_f(r.get("result_pct")) or 0.0 for r in group]
    wins = [p for p in pcts if p > 0]
    losses = [p for p in pcts if p < 0]
    gw, gl = sum(wins), -sum(losses)
    mfes = [_f((r.get("score_breakdown") or {}).get("mfe_pct")) for r in group]
    mfes = [m for m in mfes if m is not None]
    return {
        "n": n,
        "win_pct": round(100 * len(wins) / n, 1),
        "exp": round(sum(pcts) / n, 3),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "pf": round(gw / gl, 2) if gl else (9.99 if gw else 0.0),
        "total": round(sum(pcts), 1),
        "avg_mfe": round(sum(mfes) / len(mfes), 2) if mfes else 0.0,
    }


def _rode(r: dict) -> bool:
    return bool((r.get("score_breakdown") or {}).get("trend_ride_ever"))


def _by_reason(rows: list[dict]) -> dict:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r.get("closed_reason") or "unknown", []).append(r)
    return {k: _stats(v) for k, v in sorted(out.items(), key=lambda kv: -len(kv[1]))}


def summarize(rows: list[dict]) -> dict:
    """Pure aggregation over CLOSED signal rows (any list; non-swings are ignored)."""
    graded = [r for r in rows
              if r.get("result") in ("win", "loss") and r.get("result_pct") is not None]
    swings = [r for r in graded if trend_ride.is_swing(r)]
    rode = [r for r in swings if _rode(r)]
    did_not = [r for r in swings if not _rode(r)]

    # The "gave it back" failure mode among rides.
    gaveback = []
    for r in rode:
        mfe = _f((r.get("score_breakdown") or {}).get("mfe_pct"))
        res = _f(r.get("result_pct"))
        if mfe is not None and res is not None and mfe >= _GAVEBACK_MFE_MIN and res <= _GAVEBACK_RESULT_MAX:
            gaveback.append(r)

    # Early exits on swings that did NOT ride — the baseline trend_ride should beat /
    # shrink. (A riding swing suppresses these, so this set should trend toward 0.)
    early_no_ride = [r for r in did_not if (r.get("closed_reason") in _EARLY_EXITS)]

    return {
        "rode":               _stats(rode),
        "did_not_ride":       _stats(did_not),
        "rode_by_reason":     _by_reason(rode),
        "did_not_by_reason":  _by_reason(did_not),
        "trend_break":        _stats([r for r in rode if r.get("closed_reason") == "trend_break"]),
        "early_exit_baseline": _stats(early_no_ride),   # structure_reversal + market_close, non-riders
        "gave_back": {
            "n": len(gaveback),
            "tickers": [r.get("ticker") for r in gaveback][:20],
            "criteria": f"mfe_pct>={_GAVEBACK_MFE_MIN}% but result_pct<={_GAVEBACK_RESULT_MAX}%",
        },
        "counts": {
            "swings_total": len(swings),
            "rode": len(rode),
            "did_not_ride": len(did_not),
            "structure_reversal_on_swings": len([r for r in swings if r.get("closed_reason") == "structure_reversal"]),
        },
    }


def build(sb, days: int = 30) -> dict:
    """Fetch CLOSED signals from the last `days` and summarize. Read-only."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows: list[dict] = []
    for off in range(0, 8000, 1000):
        chunk = (sb.table("signals")
                 .select("ticker,direction,strategy_type,setup_type,timeframe,result,"
                         "result_pct,closed_reason,closed_at,score_breakdown")
                 .eq("status", "closed").gte("created_at", since)
                 .order("created_at").range(off, off + 999).execute().data) or []
        rows += chunk
        if len(chunk) < 1000:
            break
    out = summarize(rows)
    out["window_days"] = days
    out["closed_pulled"] = len(rows)
    return out


def _fmt(s: dict) -> str:
    return (f"n={s['n']:<4} win={s['win_pct']:>5.1f}%  exp={s['exp']:>6.2f}%  "
            f"avgW={s['avg_win']:>5.2f} avgL={s['avg_loss']:>6.2f}  PF={s['pf']:>4.2f}  "
            f"mfe={s['avg_mfe']:>5.2f}  net={s['total']:>6.1f}%")


def render(sc: dict) -> str:
    L = []
    L.append(f"TREND-RIDE SCORECARD — last {sc.get('window_days')}d ({sc.get('closed_pulled')} closed pulled)")
    c = sc["counts"]
    L.append(f"  swings={c['swings_total']}  rode={c['rode']}  did_not_ride={c['did_not_ride']}  "
             f"structure_reversal_on_swings={c['structure_reversal_on_swings']}")
    L.append("")
    L.append(f"  RODE            {_fmt(sc['rode'])}")
    L.append(f"  DID NOT RIDE    {_fmt(sc['did_not_ride'])}")
    L.append(f"  trend_break     {_fmt(sc['trend_break'])}   (the new ride-exit)")
    L.append(f"  early baseline  {_fmt(sc['early_exit_baseline'])}   (structure_reversal+market_close, non-riders)")
    gb = sc["gave_back"]
    L.append(f"  GAVE IT BACK    n={gb['n']}  ({gb['criteria']})  {gb['tickers']}")
    L.append("")
    L.append("  rode by exit reason:")
    for k, s in sc["rode_by_reason"].items():
        L.append(f"    {k:20} {_fmt(s)}")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv(override=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    _sb = create_client(os.environ["SUPABASE_URL"],
                        os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"])
    print(render(build(_sb, args.days)))
