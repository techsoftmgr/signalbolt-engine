"""
Historical backtest — run each detector's ENTRY logic over YEARS of real Alpaca
SIP daily bars, feed the entries into the replay engine with a realistic cost
model, and produce a per-detector × per-regime, walk-forward verdict.

This is the DISCOVERY tool: instead of waiting months for live trades to dribble
in (slow + confounded by engine changes), replay the frozen rules over history
to get thousands of trades across every regime in minutes.

⚠️ FIDELITY: each predicate here is an APPROXIMATION of the live detector's entry
condition. A backtest is only as honest as its predicate, so detectors are added
+ VALIDATED against live signals incrementally — NOT all at once. Tier-3
detectors (OPTIONS_FLOW / DARK_POOL / DEEP_VALUE) CANNOT be backtested (no
historical third-party data) and are forward-test-only.

Predicates are PURE (a df window in → direction|None) → unit-tested. `run()` does
the I/O (fetch + loop + aggregate). No look-ahead: a predicate sees only bars up
to and including the entry bar; exits walk the bars AFTER it.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("signalbolt.hist_backtest")

_WARMUP = 60        # bars of history before the first possible entry
_COOLDOWN = 8       # don't re-fire the same (ticker, detector) within N bars
_DEFAULT_COST = 0.15  # round-trip % (spread + slippage + commission estimate)


# ── detector entry predicates (pure; win = df rows UP TO + INCLUDING entry bar) ──
# Each returns "LONG" / "SHORT" / None. df columns: open, high, low, close, volume.

def _breakout(win):
    if len(win) < 21:
        return None
    today = win.iloc[-1]; prior = win.iloc[-21:-1]
    if today["close"] >= prior["high"].max() and today["volume"] >= 1.5 * prior["volume"].mean():
        return "LONG"
    return None


def _breakdown(win):
    if len(win) < 21:
        return None
    today = win.iloc[-1]; prior = win.iloc[-21:-1]
    if today["close"] <= prior["low"].min() and today["volume"] >= 1.5 * prior["volume"].mean():
        return "SHORT"
    return None


def _accum_forming(win):
    # heavy up-volume, NOT yet at the 20-day high (volume leads price)
    if len(win) < 21:
        return None
    today = win.iloc[-1]; prior = win.iloc[-21:-1]
    hi = prior["high"].max()
    if (today["close"] > today["open"] and today["volume"] >= 1.8 * prior["volume"].mean()
            and today["close"] < hi * 0.985):
        return "LONG"
    return None


def _distrib_forming(win):
    # heavy down-volume while still above the 20-day average (distribution into strength)
    if len(win) < 21:
        return None
    today = win.iloc[-1]; prior = win.iloc[-20:]
    ma = prior["close"].mean()
    if (today["close"] < today["open"] and today["volume"] >= 1.8 * prior["volume"].mean()
            and today["close"] > ma):
        return "SHORT"
    return None


def _compression(win):
    # squeeze: prior bar's 14-bar ATR% at a 40-bar low, then today expands/breaks
    if len(win) < 45:
        return None
    import numpy as np
    h, l, c = win["high"].values, win["low"].values, win["close"].values
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    atr = tr[-15:-1].mean()
    atr_hist = tr[-44:-1]
    if atr_hist.size == 0 or atr > np.percentile(atr_hist, 25):
        return None  # not compressed
    today = win.iloc[-1]; prior = win.iloc[-11:-1]
    if today["close"] >= prior["high"].max():
        return "LONG"
    if today["close"] <= prior["low"].min():
        return "SHORT"
    return None


# Tier-1 single-name predicates registered so far (extended + validated over time)
DETECTORS = {
    "BREAKOUT":        _breakout,
    "BREAKDOWN":       _breakdown,
    "ACCUM_FORMING":   _accum_forming,
    "DISTRIB_FORMING": _distrib_forming,
    "COMPRESSION":     _compression,
}

# Documented but NOT YET registered (fidelity work pending) / un-backtestable:
PENDING = ["BREAKOUT_FORMING", "BREAKDOWN_FORMING", "PEAK", "TURNAROUND",
           "PULLBACK", "SWING_BREAKOUT", "TREND_MOMENTUM (cross-sectional)"]
CANNOT_BACKTEST = ["OPTIONS_FLOW", "DARK_POOL", "DEEP_VALUE",
                   "EMA_RECLAIM (intraday)", "GAP_ENGINE (intraday)"]


# ── historical regime (proxy from SPY: trend × realized vol) ──

def _spy_regime(spy_df):
    """date(ISO) → RISK_ON / NEUTRAL / RISK_OFF, from SPY vs 200-SMA + 20d vol."""
    import numpy as np
    out = {}
    if spy_df is None or len(spy_df) < 200:
        return out
    close = spy_df["close"]
    sma200 = close.rolling(200).mean()
    ret = close.pct_change()
    vol = ret.rolling(20).std()
    vmed = vol.median()
    for i in range(len(spy_df)):
        d = str(spy_df.index[i].date() if hasattr(spy_df.index[i], "date") else spy_df.index[i])[:10]
        s, v = sma200.iloc[i], vol.iloc[i]
        c = close.iloc[i]
        if np.isnan(s) or np.isnan(v):
            out[d] = "NEUTRAL"; continue
        if c < s:
            out[d] = "RISK_OFF"
        elif v > vmed * 1.3:
            out[d] = "RISK_OFF" if c < s else "NEUTRAL"
        else:
            out[d] = "RISK_ON"
    return out


def _regime_allows(direction: str, regime: str) -> bool:
    """Mirror the LIVE regime-alignment gate: no LONG in a risk-off bucket,
    no SHORT in a risk-on bucket. NEUTRAL allows both."""
    if direction == "LONG":
        return regime != "RISK_OFF"
    return regime != "RISK_ON"


def run(universe: list, years: int = 3, hold_days: int = 10,
        exit_params: dict | None = None, cost_pct: float = _DEFAULT_COST,
        regime_gate: bool = False) -> dict:
    """Replay every registered detector over `years` of daily bars for `universe`.
    Returns per (detector × regime) expectancy, edge-vs-SPY, walk-forward.

    regime_gate=True applies the LIVE regime-alignment filter (only count a trade
    in regimes the live engine would actually allow) for an apples-to-apples
    comparison. Never raises (best-effort)."""
    try:
        from engine.alpaca_client import get_bars
        from engine.replay_backtest import replay
        from collections import defaultdict

        exit_params = exit_params or {"stop_pct": 8, "target_pct": 12, "trail_pct": None}
        days = int(years * 365 + _WARMUP + 10)
        spy = get_bars("SPY", "1Day", days)
        regime = _spy_regime(spy) if spy is not None else {}
        spy_close = {str(spy.index[i].date())[:10]: float(spy["close"].iloc[i])
                     for i in range(len(spy))} if spy is not None else {}
        spy_dates = sorted(spy_close)

        trades = []   # {detector, direction, regime, net, edge, date}
        scanned = 0
        for tk in universe:
            df = get_bars(tk, "1Day", days)
            if df is None or len(df) < _WARMUP + hold_days + 5:
                continue
            last_fire = {}   # detector -> last bar index fired (cooldown)
            for i in range(_WARMUP, len(df) - hold_days - 1):
                win = df.iloc[: i + 1]
                entry_px = float(df["close"].iloc[i])
                if entry_px <= 0:
                    continue
                edate = str(df.index[i].date())[:10]
                rgm = regime.get(edate, "NEUTRAL")
                fwd = df.iloc[i + 1: i + 1 + hold_days]
                fwd_bars = [(float(r.high), float(r.low), float(r.close))
                            for r in fwd.itertuples()]
                if not fwd_bars:
                    continue
                scanned += 1
                for name, pred in DETECTORS.items():
                    if i - last_fire.get(name, -999) < _COOLDOWN:
                        continue
                    try:
                        d = pred(win)
                    except Exception:
                        d = None
                    if not d:
                        continue
                    if regime_gate and not _regime_allows(d, rgm):
                        continue   # live engine would have blocked this regime/direction
                    last_fire[name] = i
                    out = replay(d, entry_px, fwd_bars, {**exit_params, "cost_pct": cost_pct})
                    if out.get("outcome") is None:
                        continue
                    net = out["realized_net_pct"]
                    # SPY return over the same hold (direction-adjusted benchmark)
                    spy_entry = spy_close.get(edate)
                    spy_exit = None
                    if spy_entry:
                        fut = [x for x in spy_dates if x > edate][:hold_days]
                        if fut:
                            spy_exit = spy_close.get(fut[-1])
                    edge = None
                    if spy_entry and spy_exit:
                        sret = (spy_exit - spy_entry) / spy_entry * 100
                        bench = sret if d == "LONG" else -sret
                        edge = round(net - bench, 3)
                    trades.append({"detector": name, "direction": d, "regime": rgm,
                                   "net": net, "edge": edge, "date": edate})

        # ── aggregate ──
        def _agg(rows):
            n = len(rows)
            if not n:
                return {"n": 0}
            nets = [r["net"] for r in rows]
            edges = [r["edge"] for r in rows if r["edge"] is not None]
            wins = sum(1 for x in nets if x > 0)
            return {"n": n, "exp_net": round(sum(nets) / n, 3),
                    "win_rate": round(100 * wins / n, 1),
                    "edge_vs_spy": round(sum(edges) / len(edges), 3) if edges else None,
                    "total": round(sum(nets), 1)}

        by = defaultdict(list)
        for t in trades:
            by[(t["detector"], t["regime"])].append(t)
            by[(t["detector"], "ALL")].append(t)
        # walk-forward: split each detector's trades by date (70/30)
        wf = {}
        bydet = defaultdict(list)
        for t in trades:
            bydet[t["detector"]].append(t)
        for det, rows in bydet.items():
            rows.sort(key=lambda r: r["date"])
            k = int(len(rows) * 0.7)
            wf[det] = {"in_sample": _agg(rows[:k]), "out_sample": _agg(rows[k:])}

        segments = [{"detector": d, "regime": r, **_agg(rows)}
                    for (d, r), rows in sorted(by.items())]
        return {
            "years": years, "hold_days": hold_days, "cost_pct": cost_pct,
            "exit_params": exit_params, "universe_n": len(universe),
            "bars_scanned": scanned, "total_trades": len(trades),
            "registered": list(DETECTORS), "pending": PENDING,
            "cannot_backtest": CANNOT_BACKTEST,
            "by_detector_regime": segments, "walk_forward": wf,
        }
    except Exception as e:
        logger.error(f"[hist_backtest] run failed: {e}")
        return {"error": str(e), "total_trades": 0}
