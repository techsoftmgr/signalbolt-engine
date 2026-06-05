"""
Daily EOD performance snapshot — one immutable row per trading day in
`daily_performance`. The synthesis layer over the per-signal MFE capture
(score_breakdown.mfe_pct) + the regime timeline (regime_history): records the
day's CLOSED outcomes (by detector / conviction / direction), profit give-back
(peak vs realized), the regime path, and the ACTIVE-book state.

Written ~8:05 PM ET (after the full 4 AM–8 PM extended session so the MFE peaks
are complete). Realized closes are identical to a 4 PM run — the engine closes
positions only during RTH + the momentum monitor at 4:25 PM — but the 8 PM run
captures the full-day MFE/give-back incl. after-hours (the KORU lesson).

`_aggregate()` is PURE (given rows + prices) → unit-tested. `compute_and_store()`
fetches + upserts. Best-effort throughout — never raises into the scheduler.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger("signalbolt.daily_performance")
ET = ZoneInfo("America/New_York")
_NEAR_PCT = 1.5   # active signal "near" a level when within this % of stop/target


def _tier(cs) -> str:
    try:
        cs = float(cs)
    except (TypeError, ValueError):
        return "?"
    if cs >= 90: return "A+ (90+)"
    if cs >= 80: return "A (80-89)"
    if cs >= 70: return "B+ (70-79)"
    if cs >= 60: return "B (60-69)"
    return "C (<60)"


def _det(r: dict) -> str:
    return (r.get("score_breakdown") or {}).get("detector_source") or r.get("strategy_type") or "NA"


def _grp_stats(rows: list) -> dict:
    n = len(rows)
    pcts = [float(r["result_pct"]) for r in rows]
    wins = sum(1 for p in pcts if p > 0)
    net = sum(pcts)
    return {"n": n, "wins": wins, "net": round(net, 2),
            "avg": round(net / n, 2) if n else None,
            "win_rate": round(100 * wins / n, 1) if n else None}


def _aggregate(closed: list, active: list, prices: dict, regime_rows: list, trade_date) -> dict:
    """Pure: build the daily_performance row from already-fetched inputs."""
    from collections import defaultdict

    # ── market context ──
    rg = sorted(regime_rows or [], key=lambda x: x.get("captured_at") or "")
    regime_close = rg[-1].get("regime_type") if rg else None
    vix = rg[-1].get("vix") if rg else None
    path = " > ".join(f"{x.get('session','?')} {x.get('regime_type','?')}" for x in rg) or None

    # ── CLOSED today (realized) ──
    cl = [r for r in (closed or []) if r.get("result_pct") is not None]
    base = _grp_stats(cl)
    gw = sum(float(r["result_pct"]) for r in cl if float(r["result_pct"]) > 0)
    gl = -sum(float(r["result_pct"]) for r in cl if float(r["result_pct"]) < 0)
    pf = round(gw / gl, 2) if gl > 0 else (99.0 if gw > 0 else None)

    def _is_carried(r):
        try:
            o = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")).astimezone(ET).date()
            return o < trade_date
        except Exception:
            return False
    carried = sum(1 for r in cl if _is_carried(r))

    longs  = [r for r in cl if (r.get("direction") or "").upper() == "LONG"]
    shorts = [r for r in cl if (r.get("direction") or "").upper() == "SHORT"]
    ls, ss = _grp_stats(longs), _grp_stats(shorts)

    # give-back = sum of (peak MFE − realized) over closed, where MFE was captured
    giveback = 0.0
    for r in cl:
        mfe = (r.get("score_breakdown") or {}).get("mfe_pct")
        if mfe is not None:
            giveback += max(0.0, float(mfe) - float(r["result_pct"]))

    byd = defaultdict(list)
    byt = defaultdict(list)
    for r in cl:
        byd[_det(r)].append(r)
        byt[_tier(r.get("confidence_score"))].append(r)
    by_detector   = {k: _grp_stats(v) for k, v in byd.items()}
    by_conviction = {k: _grp_stats(v) for k, v in byt.items()}

    def _mover(r):
        return {"ticker": r.get("ticker"), "detector": _det(r),
                "direction": r.get("direction"), "pct": round(float(r["result_pct"]), 2)}
    top_winner = _mover(max(cl, key=lambda r: float(r["result_pct"]))) if cl else None
    top_loser  = _mover(min(cl, key=lambda r: float(r["result_pct"]))) if cl else None

    # ── ACTIVE book snapshot ──
    act = active or []
    a_long = sum(1 for r in act if (r.get("direction") or "").upper() == "LONG")
    a_short = len(act) - a_long
    unreal_sum = 0.0; unreal_count = 0; near = 0; a_giveback = 0.0
    for r in act:
        cur = prices.get(r.get("ticker"))
        try:
            e = float(r.get("entry_price") or 0)
        except (TypeError, ValueError):
            e = 0
        if cur and e > 0:
            is_long = (r.get("direction") or "").upper() == "LONG"
            u = ((cur - e) / e * 100) if is_long else ((e - cur) / e * 100)
            unreal_sum += u; unreal_count += 1
            mfe = (r.get("score_breakdown") or {}).get("mfe_pct")
            if mfe is not None:
                a_giveback += max(0.0, float(mfe) - u)
            for lvl in (r.get("stop_loss"), r.get("target_one")):
                try:
                    if lvl and abs(cur - float(lvl)) / cur * 100 <= _NEAR_PCT:
                        near += 1; break
                except (TypeError, ValueError):
                    pass

    return {
        "trade_date": str(trade_date),
        "regime_close": regime_close, "regime_path": path, "vix": vix,
        "closed_n": base["n"], "closed_wins": base["wins"], "closed_win_rate": base["win_rate"],
        "closed_net_pct": base["net"], "closed_avg_pct": base["avg"], "closed_profit_factor": pf,
        "carried_n": carried,
        "long_n": ls["n"], "long_win_rate": ls["win_rate"], "long_net_pct": ls["net"],
        "short_n": ss["n"], "short_win_rate": ss["win_rate"], "short_net_pct": ss["net"],
        "giveback_pct": round(giveback, 2),
        "by_detector": by_detector, "by_conviction": by_conviction,
        "top_winner": top_winner, "top_loser": top_loser,
        "active_n": len(act),
        "active_net_unreal_pct": round(unreal_sum, 2) if unreal_count else None,
        "active_long_n": a_long, "active_short_n": a_short,
        "active_near_levels": near, "active_giveback_pct": round(a_giveback, 2) if act else None,
    }


def compute_and_store(sb) -> dict | None:
    """Fetch today's closed + active signals + regime + active prices, build the
    snapshot, and UPSERT the daily_performance row (idempotent per trade_date).
    Never raises."""
    try:
        et_now = datetime.now(ET)
        trade_date = et_now.date()
        start_utc = datetime(trade_date.year, trade_date.month, trade_date.day, tzinfo=ET).astimezone(timezone.utc)
        si = start_utc.isoformat()

        closed = (sb.table("signals")
                  .select("ticker,direction,result,result_pct,confidence_score,score_breakdown,strategy_type,created_at,closed_at")
                  .not_.is_("closed_at", "null").gte("closed_at", si)
                  .neq("strategy_type", "deep_value").limit(2000).execute().data) or []
        active = (sb.table("signals")
                  .select("ticker,direction,entry_price,stop_loss,target_one,confidence_score,score_breakdown,strategy_type")
                  .eq("status", "active").neq("strategy_type", "deep_value").limit(2000).execute().data) or []
        try:
            regime_rows = (sb.table("regime_history").select("regime_type,session,vix,captured_at")
                           .gte("captured_at", si).order("captured_at").execute().data) or []
        except Exception:
            regime_rows = []

        prices: dict = {}
        tickers = sorted({r["ticker"] for r in active if r.get("ticker")})
        if tickers:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockLatestTradeRequest
                import os
                cl = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
                feed = os.environ.get("ALPACA_DATA_FEED", "sip")
                tr = cl.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=tickers, feed=feed))
                prices = {k: float(v.price) for k, v in tr.items()}
            except Exception as e:
                logger.debug(f"[daily_perf] price fetch failed: {e}")

        row = _aggregate(closed, active, prices, regime_rows, trade_date)
        sb.table("daily_performance").upsert(row, on_conflict="trade_date").execute()
        logger.info(f"[daily_perf] {trade_date}: closed {row['closed_n']} "
                    f"(win {row['closed_win_rate']}%, net {row['closed_net_pct']}%), "
                    f"active {row['active_n']}, giveback {row['giveback_pct']}%")
        return row
    except Exception as e:
        logger.error(f"[daily_perf] compute_and_store failed: {e}")
        return None
