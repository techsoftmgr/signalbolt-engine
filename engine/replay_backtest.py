"""
Real-bar replay backtester — the keystone for learning SL/TP + exit timing.

Takes a closed signal, pulls its ACTUAL Alpaca SIP forward bars from entry, and
walks them bar-by-bar under a candidate exit policy (stop / target / trail /
breakeven / time-stop) to get the realized outcome. No yfinance, no synthetic
data — real SIP tape only.

`replay()` is PURE (given bars + params) → unit-tested, NO look-ahead: the stop/
target in force on bar i were set by bars < i; trailing/breakeven update only
AFTER the bar's exit checks. Same-bar stop+target → assume STOP first
(conservative, matches gate_validator).

Powers: A/B a candidate SL/TP vs as-traded (`run_param_set`), and the
exit_optimizer's per-(detector×regime) parameter search.

Params (dict, all distances are % of entry; all optional except stop):
  stop_pct          initial stop distance
  target_pct        fixed take-profit (None → no fixed target, ride the trail)
  trail_pct         trailing stop: lock to (peak_favorable − trail_pct)
  breakeven_at_pct  once favorable ≥ this, raise stop to entry (0% P&L)
  time_stop_bars    exit at close after N bars
  cost_pct          round-trip cost subtracted from realized (default 0.10)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.replay")

_DEFAULT_COST = 0.10


def replay(direction: str, entry: float, bars: list, params: dict) -> dict:
    """Pure bar-walk. `bars` = iterable of (high, low, close). Returns the
    realized outcome under `params`. Direction-aware (LONG/SHORT)."""
    try:
        entry = float(entry)
    except (TypeError, ValueError):
        return {"outcome": None, "exit_reason": "bad_entry"}
    if entry <= 0 or not bars:
        return {"outcome": None, "exit_reason": "no_data"}

    is_long = (direction or "").upper() == "LONG"
    stop_pct   = float(params.get("stop_pct") or 0) or None
    target_pct = params.get("target_pct")
    trail_pct  = params.get("trail_pct")
    be_at      = params.get("breakeven_at_pct")
    time_stop  = params.get("time_stop_bars")
    cost_pct   = float(params.get("cost_pct", _DEFAULT_COST))
    target_pct = float(target_pct) if target_pct is not None else None
    trail_pct  = float(trail_pct) if trail_pct is not None else None
    be_at      = float(be_at) if be_at is not None else None
    # MACD profit-lock: once favorable >= macd_lock_arm %, exit at close when the
    # MACD histogram flips AGAINST the position (the live signal_advisor's intent).
    macd_hist  = params.get("macd_hist")
    macd_arm   = params.get("macd_lock_arm")
    macd_arm   = float(macd_arm) if macd_arm is not None else None

    def pnl(price):
        return ((price - entry) if is_long else (entry - price)) / entry * 100.0

    stop_pnl = -stop_pct if stop_pct else None     # adverse threshold in P&L %
    peak_pnl = 0.0
    mfe = 0.0
    mae = 0.0
    realized = None
    reason = "end"
    held = 0

    for i, bar in enumerate(bars):
        try:
            high, low, close = float(bar[0]), float(bar[1]), float(bar[2])
        except (TypeError, ValueError, IndexError):
            continue
        held = i + 1
        fav_price = high if is_long else low       # best-case price this bar
        adv_price = low if is_long else high       # worst-case price this bar
        fav_pnl, adv_pnl = pnl(fav_price), pnl(adv_price)

        # ── exit checks vs stop/target set by PRIOR bars (no look-ahead) ──
        hit_stop   = stop_pnl is not None and adv_pnl <= stop_pnl
        hit_target = target_pct is not None and fav_pnl >= target_pct
        if hit_stop or hit_target:
            mfe = max(mfe, fav_pnl); mae = min(mae, adv_pnl)
            if hit_stop:                            # conservative: stop wins ties
                realized, reason = stop_pnl, ("stop_and_target" if hit_target else "stop")
            else:
                realized, reason = target_pct, "target"
            break

        # ── no exit: book excursions, then advance stop (breakeven/trail) ──
        mfe = max(mfe, fav_pnl); mae = min(mae, adv_pnl)
        peak_pnl = max(peak_pnl, fav_pnl)
        if be_at is not None and peak_pnl >= be_at:
            stop_pnl = 0.0 if stop_pnl is None else max(stop_pnl, 0.0)
        if trail_pct is not None:
            trailed = peak_pnl - trail_pct
            stop_pnl = trailed if stop_pnl is None else max(stop_pnl, trailed)
        if (macd_arm is not None and macd_hist is not None and i > 0
                and i < len(macd_hist)):
            cp = pnl(close)
            mp, mn = macd_hist[i - 1], macd_hist[i]
            if cp >= macd_arm and mp is not None and mn is not None:
                flipped = (mp > 0 >= mn) if is_long else (mp < 0 <= mn)
                if flipped:                          # momentum turned against us in profit
                    realized, reason = cp, "macd_lock"
                    break
        if time_stop is not None and held >= int(time_stop):
            realized, reason = pnl(close), "time"
            break

    if realized is None:
        # never exited — mark at the last close
        try:
            realized = pnl(float(bars[-1][2]))
        except Exception:
            realized = 0.0
        reason = "end"

    realized = round(realized, 3)
    return {
        "outcome": "win" if realized > 0 else "loss",
        "exit_reason": reason,
        "realized_pct": realized,
        "realized_net_pct": round(realized - cost_pct, 3),
        "bars_held": held,
        "mfe_pct": round(mfe, 2),
        "mae_pct": round(mae, 2),
    }


# ── real-bar fetch + signal replay (I/O) ──────────────────────────────────────

def _to_dt(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def fetch_forward_bars(ticker: str, entry_iso: str, horizon_days: int = 7,
                       timeframe: str = "15Min") -> list:
    """REAL Alpaca SIP bars from a signal's entry forward, as (high, low, close)
    tuples. [] on any failure."""
    try:
        from engine.alpaca_client import get_bars
        entry_dt = _to_dt(entry_iso)
        if entry_dt is None:
            return []
        if entry_dt.tzinfo is None:
            entry_dt = entry_dt.replace(tzinfo=timezone.utc)
        days_back = (datetime.now(timezone.utc) - entry_dt).days + int(horizon_days) + 2
        df = get_bars(ticker, timeframe, max(2, days_back))
        if df is None or df.empty:
            return []
        fwd = df[df.index >= entry_dt]
        out = [(float(r["high"]), float(r["low"]), float(r["close"]))
               for _, r in fwd.iterrows()]
        return out
    except Exception as e:
        logger.debug(f"[replay] fetch_forward_bars({ticker}) failed: {e}")
        return []


def replay_signal(signal: dict, params: dict, horizon_days: int = 7,
                  timeframe: str = "15Min") -> dict | None:
    """Replay one closed signal under `params` on its real forward bars."""
    tk = signal.get("ticker")
    entry = signal.get("entry_price")
    start = signal.get("created_at")
    if not tk or entry is None or not start:
        return None
    bars = fetch_forward_bars(tk, start, horizon_days, timeframe)
    if not bars:
        return None
    res = replay(signal.get("direction"), entry, bars, params)
    res["ticker"] = tk
    res["as_traded_pct"] = signal.get("result_pct")
    return res


def run_param_set(sb, params: dict, days: int = 45, detector: str | None = None,
                  horizon_days: int = 7, timeframe: str = "15Min",
                  limit: int = 400) -> dict:
    """Replay every closed signal in the window under `params` → aggregate vs the
    AS-TRADED baseline. The A/B tool for a candidate SL/TP. Never raises."""
    try:
        from datetime import timedelta
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        # Order by recency + fetch a generous cap BEFORE the (post-query) detector
        # filter, so a rare detector's signals aren't dropped by an arbitrary slice.
        rows = (sb.table("signals")
                .select("ticker,direction,entry_price,result_pct,created_at,score_breakdown,strategy_type")
                .eq("status", "closed").gte("created_at", since)
                .neq("strategy_type", "deep_value")
                .order("created_at", desc=True).limit(2000).execute().data) or []
        if detector:
            rows = [r for r in rows
                    if ((r.get("score_breakdown") or {}).get("detector_source") == detector
                        or r.get("strategy_type") == detector)]
        rows = rows[:limit]
        rep, base = [], []
        for r in rows:
            out = replay_signal(r, params, horizon_days, timeframe)
            if not out:
                continue
            rep.append(out["realized_net_pct"])
            if r.get("result_pct") is not None:
                base.append(float(r["result_pct"]))
        n = len(rep)
        def _stat(xs):
            if not xs:
                return {"n": 0}
            wins = sum(1 for x in xs if x > 0)
            return {"n": len(xs), "expectancy": round(sum(xs) / len(xs), 3),
                    "total": round(sum(xs), 2), "win_rate": round(100 * wins / len(xs), 1)}
        return {"params": params, "days": days, "detector": detector,
                "replayed": _stat(rep), "as_traded": _stat(base),
                "note": "replay uses real SIP bars; as_traded is the engine's actual exit"}
    except Exception as e:
        logger.error(f"[replay] run_param_set failed: {e}")
        return {"params": params, "replayed": {"n": 0}, "error": str(e)}
