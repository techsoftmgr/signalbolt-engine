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


def _rsi(closes, period=14):
    import numpy as np
    if len(closes) < period + 1:
        return 50.0
    delta = np.diff(closes[-(period + 1):])
    gain = np.where(delta > 0, delta, 0.0).mean()
    loss = np.where(delta < 0, -delta, 0.0).mean()
    if loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + gain / loss)


def _breakout_forming(win):
    # LONG: pressing the 20-day high on volume, NOT yet broken (anticipatory)
    if len(win) < 21:
        return None
    today = win.iloc[-1]; prior = win.iloc[-21:-1]; hi = prior["high"].max()
    if (today["close"] < hi and today["close"] >= hi * 0.985
            and today["volume"] >= 1.5 * prior["volume"].mean()):
        return "LONG"
    return None


def _breakdown_forming(win):
    # SHORT: lost the 20-day average on heavy down-volume, not yet at the 20d low
    if len(win) < 21:
        return None
    today = win.iloc[-1]; prior = win.iloc[-20:]
    ma = prior["close"].mean(); lo = win.iloc[-21:-1]["low"].min()
    if (today["close"] < ma and today["close"] > lo and today["close"] < today["open"]
            and today["volume"] >= 1.5 * prior["volume"].mean()):
        return "SHORT"
    return None


def _peak(win):
    # SHORT: exhaustion top — overbought + bearish reversal near a recent high
    if len(win) < 21:
        return None
    closes = win["close"].values
    today = win.iloc[-1]; prior = win.iloc[-21:-1]; hi = prior["high"].max()
    if (_rsi(closes) > 68 and today["close"] < today["open"]
            and today["close"] < closes[-2] and today["close"] >= hi * 0.95):
        return "SHORT"
    return None


def _turnaround(win):
    # LONG: oversold reversal near a recent low (bottoming)
    if len(win) < 21:
        return None
    closes = win["close"].values
    today = win.iloc[-1]; prior = win.iloc[-21:-1]; lo = prior["low"].min()
    if (_rsi(closes) < 32 and today["close"] > today["open"]
            and today["close"] > closes[-2] and today["close"] <= lo * 1.05):
        return "LONG"
    return None


def _pullback(win):
    # LONG: pullback reclaim inside an uptrend (dip below SMA20 then back above)
    if len(win) < 55:
        return None
    import numpy as np
    closes = win["close"].values
    sma20 = np.mean(closes[-20:]); sma50 = np.mean(closes[-50:])
    today = win.iloc[-1]
    if today["close"] > sma50 and sma20 > sma50:
        if win.iloc[-5:]["low"].min() < sma20 and today["close"] > sma20:
            return "LONG"
    return None


def _swing_breakout(win):
    # LONG: break of a 50-day swing high on volume (longer-horizon breakout)
    if len(win) < 51:
        return None
    today = win.iloc[-1]; prior = win.iloc[-51:-1]
    if today["close"] >= prior["high"].max() and today["volume"] >= 1.4 * prior["volume"].mean():
        return "LONG"
    return None


# Tier-1 single-name predicates (approximations of the live detectors)
DETECTORS = {
    "BREAKOUT":          _breakout,
    "BREAKDOWN":         _breakdown,
    "ACCUM_FORMING":     _accum_forming,
    "DISTRIB_FORMING":   _distrib_forming,
    "COMPRESSION":       _compression,
    "BREAKOUT_FORMING":  _breakout_forming,
    "BREAKDOWN_FORMING": _breakdown_forming,
    "PEAK":              _peak,
    "TURNAROUND":        _turnaround,
    "PULLBACK":          _pullback,
    "SWING_BREAKOUT":    _swing_breakout,
}

# All Tier-1 single-name + TREND_MOMENTUM now registered. Tier-2/3 below.
PENDING = []   # TREND_MOMENTUM via _momentum_pass; Tier-1 all registered
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


def _spy_edge(net, direction, edate, spy_close, spy_dates, hold_days):
    se = spy_close.get(edate)
    if not se:
        return None
    fut = [x for x in spy_dates if x > edate][:hold_days]
    if not fut:
        return None
    sx = spy_close.get(fut[-1])
    if not sx:
        return None
    sret = (sx - se) / se * 100
    bench = sret if direction == "LONG" else -sret
    return round(net - bench, 3)


def _momentum_pass(bars, spy_close, spy_dates, regime, hold_days, exit_params,
                   cost_pct, regime_gate, top_n=3, rebal=10, lookback=126):
    """Cross-sectional momentum (TREND_MOMENTUM): every `rebal` bars, rank the
    universe by `lookback`-day return among uptrend names (px>SMA50>SMA200), go
    LONG the top `top_n`. The academically-standard momentum factor."""
    import numpy as np
    from engine.replay_backtest import replay
    info = {}
    for tk, df in bars.items():
        if len(df) < 200:
            continue
        info[tk] = (df, df["close"].values.astype(float),
                    df["close"].rolling(50).mean().values,
                    df["close"].rolling(200).mean().values,
                    {str(df.index[i].date())[:10]: i for i in range(len(df))})
    out = []
    for rd in spy_dates[200::rebal]:
        cands = []
        for tk, (df, closes, sma50, sma200, pos) in info.items():
            i = pos.get(rd)
            if i is None or i < lookback or i >= len(closes) - hold_days - 1:
                continue
            c, c0 = closes[i], closes[i - lookback]
            if c0 <= 0 or np.isnan(sma50[i]) or np.isnan(sma200[i]):
                continue
            if not (c > sma50[i] > sma200[i]):     # uptrend filter (long-only)
                continue
            cands.append((c / c0 - 1, tk, i))
        cands.sort(reverse=True)
        for _mom, tk, i in cands[:top_n]:
            rgm = regime.get(rd, "NEUTRAL")
            if regime_gate and not _regime_allows("LONG", rgm):
                continue
            df, closes = info[tk][0], info[tk][1]
            fwd = df.iloc[i + 1: i + 1 + hold_days]
            fwd_bars = [(float(r.high), float(r.low), float(r.close)) for r in fwd.itertuples()]
            if not fwd_bars:
                continue
            res = replay("LONG", float(closes[i]), fwd_bars, {**exit_params, "cost_pct": cost_pct})
            if res.get("outcome") is None:
                continue
            net = res["realized_net_pct"]
            out.append({"detector": "TREND_MOMENTUM", "direction": "LONG", "regime": rgm,
                        "net": net,
                        "edge": _spy_edge(net, "LONG", rd, spy_close, spy_dates, hold_days),
                        "date": rd})
    return out


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
        bars = {}
        for tk in universe:
            df = get_bars(tk, "1Day", days)
            if df is not None and len(df) >= _WARMUP + hold_days + 5:
                bars[tk] = df
        # ── single-name predicate pass ──
        for tk, df in bars.items():
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

        # ── cross-sectional momentum pass (TREND_MOMENTUM) ──
        try:
            trades += _momentum_pass(bars, spy_close, spy_dates, regime,
                                     hold_days, exit_params, cost_pct, regime_gate)
        except Exception as _me:
            logger.debug(f"[hist_backtest] momentum pass failed: {_me}")

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
