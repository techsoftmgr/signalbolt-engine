"""
Peak-regime study (research / measure-first) — should the peak detector ACT ON or ESCALATE tops
in a BULL regime? (Fix #2 from the MSFT 466→370 analysis.)

Point-in-time backtest, NO look-ahead: walk a liquid universe's daily history, run
`peak_detector.score_peak` as-of each bar with the regime of that day, record every topping
occurrence by STAGE (watch / peak) and REGIME (bull / non-bull), then measure the forward SHORT
return over `horizon` trading days. The decision:
  • peak/bull  +EV  → the confirmed peaks we already fire in bull regimes pay (keep / lean in)
  • watch/bull +EV  → the topping ZONE predicts downside in bull regimes too → escalating it
                       (loosening the bull-trap block) is worth it
  • watch/bull −EV  → those are bull-traps; the block is correctly protecting us (do NOT loosen)

Daily bars via yfinance (research tool); SPY-above-rising-50MA as the bull-regime proxy. Run:
  python -m engine.peak_regime_study --days 120 --horizon 10
"""
from __future__ import annotations

import pandas as pd

_UNIVERSE = [
    "MSFT", "NVDA", "AAPL", "META", "GOOGL", "AMZN", "TSLA", "AMD", "AVGO", "CRM",
    "NFLX", "COIN", "PLTR", "MSTR", "HOOD", "SMCI", "MRVL", "LRCX", "KLAC", "ARM",
    "SNOW", "CRWD", "PANW", "NOW", "UBER", "ABNB", "SHOP", "DDOG", "NCLH", "MELI",
]


def _bull_flags(spy: pd.DataFrame) -> pd.Series:
    """SPY above a RISING 50-day MA = bull-regime proxy (mirrors the peak gate's intent)."""
    c = spy["close"].astype(float)
    ma50 = c.rolling(50).mean()
    flags = (c > ma50) & (ma50 > ma50.shift(10))
    flags.index = [ix.date() for ix in flags.index]
    return flags


def _stats(vals: list[float]) -> dict:
    n = len(vals)
    if not n:
        return {"n": 0, "short_win_pct": 0.0, "avg_short_ret_pct": 0.0, "best": 0.0, "worst": 0.0}
    wins = [v for v in vals if v > 0]
    return {
        "n": n,
        "short_win_pct": round(100 * len(wins) / n, 1),
        "avg_short_ret_pct": round(sum(vals) / n, 2),   # >0 = shorting the signal made money
        "best": round(max(vals), 1),
        "worst": round(min(vals), 1),
    }


def run(tickers=None, days: int = 120, horizon: int = 10) -> tuple[dict, list]:
    import yfinance as yf
    from engine import peak_detector as pk
    tickers = tickers or _UNIVERSE
    spy = yf.Ticker("SPY").history(period="2y", interval="1d").rename(columns=str.lower)
    bull = _bull_flags(spy)

    buckets: dict = {}
    occ: list = []
    for tk in tickers:
        try:
            df = (yf.Ticker(tk).history(period="2y", interval="1d")
                  .rename(columns=str.lower)[["open", "high", "low", "close", "volume"]])
        except Exception:
            continue
        if df is None or len(df) < 260:
            continue
        dates = [ix.date() for ix in df.index]
        n = len(df)
        last_counted: dict = {}
        start = max(260, n - days - horizon)
        for i in range(start, n - horizon):
            d = dates[i]
            res = pk.score_peak(df.iloc[: i + 1],
                                regime_type=("TRENDING_BULL" if bool(bull.get(d, False)) else "NEUTRAL"))
            if not res or res["stage"] == "none":
                continue
            stage = res["stage"]
            key = (tk, stage)
            if key in last_counted and (i - last_counted[key]) < horizon:
                continue          # dedup: one occurrence per topping episode
            last_counted[key] = i
            fwd = (float(df["close"].iloc[i + horizon]) / float(df["close"].iloc[i]) - 1) * 100
            short_ret = -fwd      # a SHORT profits when price falls
            rg = "bull" if bool(bull.get(d, False)) else "non-bull"
            buckets.setdefault(f"{stage}/{rg}", []).append(short_ret)
            occ.append((tk, str(d), stage, rg, int(res["score"]), round(short_ret, 1)))

    out = {
        "days": days, "horizon": horizon, "tickers": len([t for t in tickers]),
        "by_stage_regime": {k: _stats(v) for k, v in sorted(buckets.items())},
    }
    return out, occ


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--show", type=int, default=0, help="print N sample occurrences")
    a = ap.parse_args()
    res, occ = run(days=a.days, horizon=a.horizon)
    print(json.dumps(res, indent=2))
    print(f"\nTotal occurrences: {len(occ)}")
    if a.show:
        print("\nsample (ticker, date, stage, regime, score, short_ret%):")
        for o in occ[: a.show]:
            print("  ", o)
