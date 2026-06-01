"""
Stage-2 offline backtest — does an RS + base-tightness filter improve breakout expectancy?
==========================================================================================

STANDALONE MEASUREMENT TOOL (not wired into the engine, no endpoint, no DB writes).
Per the owner's rule: *measure expectancy before shipping a detector, cut losers.*

HYPOTHESIS
----------
A basic "close makes a new 20-day high" breakout is a coin-flip-ish event. We
suspect that layering two cheap, well-known momentum filters on top improves the
FORWARD expectancy:

  1. Relative strength vs SPY — only take the breakout if the stock's trailing
     ~63-day (one quarter) return BEATS SPY's over the same window. Buying the
     breakout in a name that is already leading the market.
  2. Base tightness (range contraction) — only take the breakout if the ~30 bars
     BEFORE the breakout formed a TIGHT consolidation (a coiled base), not a
     loose, choppy run-up. Tight bases tend to resolve with more follow-through.

We compare UNFILTERED breakouts (all of them) vs FILTERED breakouts (RS pass AND
tightness pass) on a liquid universe over ~2 years of daily bars, measuring both
20- and 60-trading-day forward returns. If FILTERED doesn't clearly beat
UNFILTERED on expectancy with a non-trivial sample, the filter is NOT worth
graduating to a tracked detector.

DEFINITIONS (the exact knobs — documented so they can be tuned/audited)
-----------------------------------------------------------------------
  * Breakout event   : today's close > max(close) of the prior LOOKBACK_HIGH (=20)
                       bars AND the previous bar was NOT already above that level
                       (a FRESH cross, so we count the event once, not every day
                       it stays extended).
  * RS filter        : stock_ret(63) - spy_ret(63) > RS_MIN (=0.0). Trailing
                       ~63-trading-day (~1 quarter) simple return of the stock vs
                       SPY measured at the breakout bar.
  * Tightness filter : over the BASE_WINDOW (=30) bars immediately BEFORE the
                       breakout bar, (max(high) - min(low)) / breakout_close
                       <= TIGHT_MAX (=0.20). A 30-bar range under 20% of price =
                       a reasonably tight base for these (often high-beta) names.
  * Volume confirm   : breakout-bar volume >= VOL_MULT (=1.5) x the prior 20-bar
                       average volume. Reported as an EXTRA cut, not part of the
                       headline FILTERED bucket (kept separate so the RS+tightness
                       result is clean).
  * Forward return   : (close[t+H] - close[t]) / close[t] for H in (20, 60)
                       trading days. Events without a full forward window are
                       skipped (no look-ahead, no partial windows).
  * Expectancy       : win_rate * mean_win + (1 - win_rate) * mean_loss
                       (mean_loss is negative), i.e. the average forward % you'd
                       expect per event. Equivalently == mean forward return; we
                       compute it the win/loss way so the components are visible.

COSTS / CAVEATS (honest)
------------------------
  * These are GROSS forward returns on the close. No commissions, no slippage, no
    spread. UNIVERSE is large/liquid mega- and large-caps so round-trip cost is
    small (a few bps), but it is NOT zero — shave ~0.1-0.2% off every event when
    judging the edge. On microcaps this would be far worse; do not extrapolate.
  * SIP daily bars, ~2y. Survivorship: UNIVERSE is a hand-picked current list, so
    there is mild survivorship/selection bias (today's known liquid names).
  * Forward returns at fixed horizons (no stop / no target) — this measures the
    raw signal's drift, not a tradeable strategy with risk management.
  * Overlapping events on the same ticker are correlated; treat N as optimistic.

USAGE
-----
    python -m engine.stage2_backtest
    python -m engine.stage2_backtest --days 900 --tight-max 0.15 --rs-min 0.0

Self-contained CLI. `import engine.config` loads the .env so alpaca_client can auth.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Loads .env (Alpaca keys) as a side effect of import — required for alpaca_client auth.
import engine.config  # noqa: F401
from engine.alpaca_client import get_bars
from engine.momentum_detector import UNIVERSE

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("signalbolt.stage2_backtest")

# ── Tunables (documented in the module docstring) ────────────────────────────
LOOKBACK_HIGH = 20      # "20-day high" breakout lookback (trading days)
RS_WINDOW     = 63      # relative-strength trailing window (~1 quarter)
RS_MIN        = 0.0     # stock must beat SPY by at least this (fraction) over RS_WINDOW
BASE_WINDOW   = 30      # bars before the breakout used to measure base tightness
TIGHT_MAX     = 0.20    # (max(high)-min(low))/close over BASE_WINDOW must be <= this
VOL_WINDOW    = 20      # prior-bar average for volume confirmation
VOL_MULT      = 1.5     # breakout-bar volume >= this x the prior VOL_WINDOW average
HORIZONS      = (20, 60)  # forward windows (trading days) measured for every event

# Need: BASE_WINDOW + LOOKBACK_HIGH history behind the bar, RS_WINDOW behind it,
# and max(HORIZONS) ahead. Pad generously so early-history edge cases are skipped.
MIN_BARS_REQUIRED = max(BASE_WINDOW + LOOKBACK_HIGH, RS_WINDOW) + max(HORIZONS) + 5


@dataclass
class Event:
    ticker:     str
    idx:        int               # integer position of the breakout bar in the df
    rs_pass:    bool
    tight_pass: bool
    vol_pass:   bool
    fwd:        dict              # {horizon: forward_return_fraction}


# ── Data fetch ────────────────────────────────────────────────────────────────

def _fetch_daily(ticker: str, days: int, retries: int = 2) -> Optional[pd.DataFrame]:
    """Fetch daily bars, tolerating transient None responses. Returns a cleaned
    DataFrame sorted by time, or None if the data is unusable."""
    for attempt in range(retries + 1):
        df = get_bars(ticker, timeframe="1Day", days=days)
        if df is not None and not df.empty:
            df = df.sort_index()
            # Drop rows with missing OHLC (corporate-action gaps etc.)
            df = df.dropna(subset=["open", "high", "low", "close"])
            if len(df) >= MIN_BARS_REQUIRED:
                return df
            return df if len(df) > 0 else None
        if attempt < retries:
            time.sleep(0.5)
    return None


# ── Per-ticker event scan ──────────────────────────────────────────────────────

def _scan_ticker(ticker: str, df: pd.DataFrame, spy_ret63: pd.Series) -> list[Event]:
    """Walk one ticker's daily history and emit every fresh 20-day-high breakout
    with its filter pass/fail flags and forward returns.

    spy_ret63 is SPY's trailing RS_WINDOW return indexed by SPY's UTC dates; we
    align it to this ticker's breakout date with an as-of (last known) join so a
    missing-date mismatch never crashes the scan.
    """
    events: list[Event] = []
    if df is None or len(df) < MIN_BARS_REQUIRED:
        return events

    close  = df["close"].to_numpy(dtype=float)
    high   = df["high"].to_numpy(dtype=float)
    low    = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    dates  = df.index
    n      = len(close)

    # Stock trailing RS_WINDOW return at each bar (NaN where insufficient history).
    stock_ret63 = np.full(n, np.nan)
    for i in range(RS_WINDOW, n):
        base = close[i - RS_WINDOW]
        if base > 0:
            stock_ret63[i] = (close[i] - base) / base

    last_fwd = max(HORIZONS)
    # Need at least LOOKBACK_HIGH prior bars to form the high, BASE_WINDOW for the
    # base, RS_WINDOW for relative strength, and last_fwd bars ahead for forward ret.
    start = max(LOOKBACK_HIGH, BASE_WINDOW, RS_WINDOW)
    end   = n - last_fwd  # exclusive — leave a full forward window at the tail

    for i in range(start, end):
        prior_high = np.max(close[i - LOOKBACK_HIGH:i])         # max close of prior 20 bars
        if not (close[i] > prior_high):
            continue
        # FRESH cross: the previous bar must NOT have already been above its own
        # prior-20 high, otherwise we'd recount a multi-day extension.
        prev_prior_high = np.max(close[i - 1 - LOOKBACK_HIGH:i - 1])
        if close[i - 1] > prev_prior_high:
            continue

        # ── RS filter ──
        s_ret = stock_ret63[i]
        try:
            # as-of align: SPY's last known trailing-63 return on or before this date
            spy_at = spy_ret63.asof(dates[i])
        except Exception:
            spy_at = np.nan
        rs_pass = (
            not np.isnan(s_ret)
            and spy_at is not None
            and not (isinstance(spy_at, float) and np.isnan(spy_at))
            and (s_ret - float(spy_at)) > RS_MIN
        )

        # ── Tightness filter (base BEFORE the breakout bar) ──
        base_hi = np.max(high[i - BASE_WINDOW:i])
        base_lo = np.min(low[i - BASE_WINDOW:i])
        tight_ratio = (base_hi - base_lo) / close[i] if close[i] > 0 else np.inf
        tight_pass = tight_ratio <= TIGHT_MAX

        # ── Volume confirmation (extra) ──
        prior_vol = volume[i - VOL_WINDOW:i]
        avg_vol = float(np.mean(prior_vol)) if len(prior_vol) else 0.0
        vol_pass = avg_vol > 0 and volume[i] >= VOL_MULT * avg_vol

        # ── Forward returns ──
        fwd = {}
        ok = True
        for h in HORIZONS:
            j = i + h
            if j >= n or close[i] <= 0:
                ok = False
                break
            fwd[h] = (close[j] - close[i]) / close[i]
        if not ok:
            continue

        events.append(Event(ticker, i, rs_pass, tight_pass, vol_pass, fwd))

    return events


# ── Stats ────────────────────────────────────────────────────────────────────

def _bucket_stats(returns: list[float]) -> dict:
    """Compute the headline stats for a list of forward returns (fractions)."""
    arr = np.array(returns, dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "win_rate": np.nan, "mean": np.nan, "median": np.nan,
                "mean_win": np.nan, "mean_loss": np.nan, "expectancy": np.nan}
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    win_rate = len(wins) / n
    mean_win = float(np.mean(wins)) if len(wins) else 0.0
    mean_loss = float(np.mean(losses)) if len(losses) else 0.0
    expectancy = win_rate * mean_win + (1 - win_rate) * mean_loss
    return {
        "n": n,
        "win_rate": win_rate,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "mean_win": mean_win,
        "mean_loss": mean_loss,
        "expectancy": expectancy,
    }


def _fmt_pct(x: float) -> str:
    return "   n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+6.2f}%"


def _print_table(title: str, rows: list[tuple[str, dict]]) -> None:
    print(f"\n{title}")
    print("-" * 92)
    header = (f"{'bucket':<22}{'N':>6}{'win%':>8}{'mean':>9}"
              f"{'median':>9}{'mean_win':>10}{'mean_loss':>11}{'expectancy':>12}")
    print(header)
    print("-" * 92)
    for label, s in rows:
        if s["n"] == 0:
            print(f"{label:<22}{0:>6}{'   n/a':>8}{'   n/a':>9}{'   n/a':>9}"
                  f"{'   n/a':>10}{'   n/a':>11}{'   n/a':>12}")
            continue
        print(f"{label:<22}{s['n']:>6}{s['win_rate'] * 100:>7.1f}%"
              f"{_fmt_pct(s['mean']):>9}{_fmt_pct(s['median']):>9}"
              f"{_fmt_pct(s['mean_win']):>10}{_fmt_pct(s['mean_loss']):>11}"
              f"{_fmt_pct(s['expectancy']):>12}")
    print("-" * 92)


# ── Main ────────────────────────────────────────────────────────────────────

def run(days: int = 760) -> None:
    universe = [t for t in UNIVERSE if t != "SPY"]
    print("=" * 92)
    print("SignalBolt STAGE-2 OFFLINE BACKTEST — RS + base-tightness filter on 20-day breakouts")
    print("=" * 92)
    print(f"Universe: {len(universe)} tickers (+ SPY benchmark)   |   History: ~{days} calendar days")
    print(f"Breakout: close > prior {LOOKBACK_HIGH}-bar high (fresh cross)")
    print(f"RS filter: stock {RS_WINDOW}d return - SPY {RS_WINDOW}d return > {RS_MIN}")
    print(f"Tightness: ({BASE_WINDOW}-bar high-low range)/close <= {TIGHT_MAX}")
    print(f"Volume (extra): breakout vol >= {VOL_MULT}x prior {VOL_WINDOW}-bar avg")
    print(f"Forward horizons: {HORIZONS} trading days   |   GROSS returns (see cost caveat)")
    print("=" * 92)

    # ── SPY benchmark first ──
    print("\nFetching SPY benchmark ...")
    spy_df = _fetch_daily("SPY", days)
    if spy_df is None or len(spy_df) < RS_WINDOW + 5:
        print("FATAL: could not fetch enough SPY history for the RS benchmark. Aborting.")
        return
    spy_close = spy_df["close"].to_numpy(dtype=float)
    spy_ret63_vals = np.full(len(spy_close), np.nan)
    for i in range(RS_WINDOW, len(spy_close)):
        base = spy_close[i - RS_WINDOW]
        if base > 0:
            spy_ret63_vals[i] = (spy_close[i] - base) / base
    spy_ret63 = pd.Series(spy_ret63_vals, index=spy_df.index).dropna()
    print(f"  SPY: {len(spy_df)} bars, RS series ready ({len(spy_ret63)} valid points).")

    # ── Per-ticker scan ──
    all_events: list[Event] = []
    fetched = skipped = 0
    print("\nScanning universe ...")
    for k, ticker in enumerate(universe, 1):
        try:
            df = _fetch_daily(ticker, days)
            if df is None or len(df) < MIN_BARS_REQUIRED:
                skipped += 1
                logger.debug("skip %s — insufficient data (%s bars)", ticker,
                             0 if df is None else len(df))
                continue
            evs = _scan_ticker(ticker, df, spy_ret63)
            all_events.extend(evs)
            fetched += 1
            print(f"  [{k:>2}/{len(universe)}] {ticker:<6} bars={len(df):>4}  events={len(evs):>3}")
        except Exception as e:  # never let one bad ticker kill the run
            skipped += 1
            logger.warning("error scanning %s: %s", ticker, e)
            continue

    print(f"\nFetched {fetched} tickers, skipped {skipped}. Total breakout events: {len(all_events)}")
    if not all_events:
        print("No breakout events found — nothing to measure.")
        return

    # ── Bucket and report per horizon ──
    for h in HORIZONS:
        unfiltered = [e.fwd[h] for e in all_events]
        filtered   = [e.fwd[h] for e in all_events if e.rs_pass and e.tight_pass]
        rs_only     = [e.fwd[h] for e in all_events if e.rs_pass]
        tight_only  = [e.fwd[h] for e in all_events if e.tight_pass]
        filt_vol   = [e.fwd[h] for e in all_events if e.rs_pass and e.tight_pass and e.vol_pass]

        rows = [
            ("UNFILTERED (all)",     _bucket_stats(unfiltered)),
            ("RS pass only",         _bucket_stats(rs_only)),
            ("Tightness pass only",  _bucket_stats(tight_only)),
            ("FILTERED (RS+tight)",  _bucket_stats(filtered)),
            ("FILTERED + volume",    _bucket_stats(filt_vol)),
        ]
        _print_table(f"FORWARD HORIZON = {h} trading days", rows)

    # ── Honest summary (use the longer horizon as the headline) ──
    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    for h in HORIZONS:
        unf = _bucket_stats([e.fwd[h] for e in all_events])
        fil = _bucket_stats([e.fwd[h] for e in all_events if e.rs_pass and e.tight_pass])
        if unf["n"] == 0 or fil["n"] == 0:
            print(f"[{h}d] insufficient events to compare.")
            continue
        d_exp = fil["expectancy"] - unf["expectancy"]
        d_win = fil["win_rate"] - unf["win_rate"]
        verdict = "IMPROVES" if d_exp > 0 else "does NOT improve"
        print(f"[{h}d] FILTERED {verdict} expectancy: "
              f"{_fmt_pct(unf['expectancy'])} -> {_fmt_pct(fil['expectancy'])} "
              f"(delta {_fmt_pct(d_exp)}); win% {unf['win_rate']*100:.1f} -> {fil['win_rate']*100:.1f} "
              f"(delta {d_win*100:+.1f}pp); N {unf['n']} -> {fil['n']}.")
    print("\nCAVEATS: GROSS returns (no commission/slippage/spread — shave ~0.1-0.2% per event;")
    print("microcaps would be far worse, this UNIVERSE is liquid). Overlapping same-ticker events")
    print("are correlated, so treat N as optimistic. Hand-picked current universe = mild survivorship")
    print("bias. Fixed-horizon forward drift, NOT a risk-managed tradeable strategy. A filter is only")
    print("worth graduating to a tracked detector if it lifts expectancy meaningfully AND keeps a")
    print("usable sample size (a tiny FILTERED N with a big number is noise, not edge).")
    print("=" * 92)


def main() -> None:
    global RS_MIN, TIGHT_MAX, VOL_MULT
    ap = argparse.ArgumentParser(description="Stage-2 offline backtest: RS + tightness filter on 20d breakouts")
    ap.add_argument("--days", type=int, default=760, help="calendar days of daily history to fetch (~2y default)")
    ap.add_argument("--rs-min", type=float, default=RS_MIN, help="min (stock 63d ret - SPY 63d ret) to pass RS")
    ap.add_argument("--tight-max", type=float, default=TIGHT_MAX, help="max (30-bar range)/close to pass tightness")
    ap.add_argument("--vol-mult", type=float, default=VOL_MULT, help="breakout vol multiple vs prior 20-bar avg")
    args = ap.parse_args()
    RS_MIN    = args.rs_min
    TIGHT_MAX = args.tight_max
    VOL_MULT  = args.vol_mult
    run(days=args.days)


if __name__ == "__main__":
    main()
