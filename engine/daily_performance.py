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
_NEAR_PCT = 1.5     # active signal "near" a level when within this % of stop/target
_MOVE_NOTABLE = 8.0  # |%| move that makes a closed/active signal a "notable" run/dump
_AH_GIVEBACK = 10.0  # active give-back (peak MFE − current) that warrants a news check

# Compact keyword sentiment for matching a headline to a move direction (a dump
# wants a bearish headline, a run wants a bullish one). Kept local so this module
# stays decoupled from the live news-feed service.
_BULLISH_WORDS = {
    "beats", "beat", "raises", "raised", "upgrade", "upgraded", "outperforms",
    "strong", "surge", "surges", "jumps", "rallies", "record", "growth",
    "accelerates", "partnership", "wins", "awarded", "positive", "approval",
    "approves", "breakthrough", "buyback", "soars", "tops", "boosts",
}
_BEARISH_WORDS = {
    "misses", "miss", "missed", "lowers", "lowered", "downgrade", "downgraded",
    "disappoints", "weak", "plunges", "drops", "falls", "warns", "warning",
    "investigation", "recall", "loss", "losses", "cut", "cuts", "delay",
    "delayed", "lawsuit", "fine", "fined", "suspension", "negative", "rejection",
    "rejects", "concern", "slumps", "sinks", "tumbles", "halts", "probe",
}


def _news_sentiment(item: dict) -> str:
    text = ((item.get("headline") or "") + " " + (item.get("summary") or "")).lower()
    b = sum(1 for w in _BEARISH_WORDS if w in text)
    g = sum(1 for w in _BULLISH_WORDS if w in text)
    if b > g: return "bearish"
    if g > b: return "bullish"
    return "neutral"


def _select_movers(closed: list, active: list, prices: dict) -> list:
    """Pure: the notable runs/dumps worth a news reason — big realized closes +
    active positions that moved hard against (or gave back a lot)."""
    movers = []
    seen = set()
    for r in (closed or []):
        try:
            p = float(r.get("result_pct"))
        except (TypeError, ValueError):
            continue
        tk = r.get("ticker")
        if not tk or tk in seen or abs(p) < _MOVE_NOTABLE:
            continue
        seen.add(tk)
        is_long = (r.get("direction") or "").upper() == "LONG"
        stock_down = (p < 0) if is_long else (p > 0)   # which way the underlying moved
        movers.append({"ticker": tk, "kind": "closed",
                       "direction": (r.get("direction") or "").upper(),
                       "move_pct": round(p, 2), "stock_down": stock_down})
    for r in (active or []):
        tk = r.get("ticker")
        if not tk or tk in seen:
            continue
        cur = prices.get(tk)
        try:
            e = float(r.get("entry_price") or 0)
        except (TypeError, ValueError):
            e = 0
        if not cur or e <= 0:
            continue
        is_long = (r.get("direction") or "").upper() == "LONG"
        u = ((cur - e) / e * 100) if is_long else ((e - cur) / e * 100)
        mfe = (r.get("score_breakdown") or {}).get("mfe_pct")
        gb = max(0.0, float(mfe) - u) if mfe is not None else 0.0
        if u <= -_MOVE_NOTABLE or gb >= _AH_GIVEBACK:
            seen.add(tk)
            movers.append({"ticker": tk, "kind": "active",
                           "direction": (r.get("direction") or "").upper(),
                           "move_pct": round(u, 2), "giveback_pct": round(gb, 2),
                           "stock_down": cur < e})
    return movers


def _match_catalyst(mover: dict, news_items: list, trade_date) -> dict | None:
    """Pure: pick the headline that best explains a mover — same ticker, prefer a
    sentiment that matches the move direction and a story published today."""
    tk = mover.get("ticker")
    want = "bearish" if mover.get("stock_down") else "bullish"
    cands = []
    for it in (news_items or []):
        if tk not in (it.get("symbols") or []):
            continue
        created = (it.get("created_at") or "")[:10]
        sent = _news_sentiment(it)
        score = (2 if sent == want else 0) + (1 if created == str(trade_date) else 0)
        cands.append((score, created, sent, it))
    if not cands:
        return None
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
    score, created, sent, it = cands[0]
    summ = (it.get("summary") or "").strip()
    if len(summ) > 240:
        summ = summ[:237] + "..."
    return {"ticker": tk, "kind": mover.get("kind"), "direction": mover.get("direction"),
            "move_pct": mover.get("move_pct"), "headline": it.get("headline"),
            "summary": summ, "url": it.get("url"), "source": it.get("source"),
            "published_at": it.get("created_at"), "sentiment": sent}


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

        # Enrich today's closed with alpha-vs-SPY (excess over market exposure)
        # before aggregating — folded here so it adds NO new scheduled job.
        try:
            from engine import alpha
            spy_oc = alpha._spy_oc_by_date(10)
            if spy_oc:
                alpha.enrich(sb, closed, spy_oc)
        except Exception as e:
            logger.debug(f"[daily_perf] alpha enrich failed: {e}")

        row = _aggregate(closed, active, prices, regime_rows, trade_date)

        # WHY the notable movers ran/dumped — match a news headline to each big
        # mover (incl. active after-hours dumps the 8 PM run is here to catch).
        catalysts = []
        try:
            movers = _select_movers(closed, active, prices)
            if movers:
                from engine.alpaca_client import get_multi_news
                news = get_multi_news([m["ticker"] for m in movers][:25], limit=50)
                for m in movers:
                    c = _match_catalyst(m, news, trade_date)
                    if c:
                        catalysts.append(c)
        except Exception as e:
            logger.debug(f"[daily_perf] catalyst enrich failed: {e}")
        row["catalysts"] = catalysts

        sb.table("daily_performance").upsert(row, on_conflict="trade_date").execute()
        logger.info(f"[daily_perf] {trade_date}: closed {row['closed_n']} "
                    f"(win {row['closed_win_rate']}%, net {row['closed_net_pct']}%), "
                    f"active {row['active_n']}, giveback {row['giveback_pct']}%, "
                    f"catalysts {len(catalysts)}")
        return row
    except Exception as e:
        logger.error(f"[daily_perf] compute_and_store failed: {e}")
        return None
