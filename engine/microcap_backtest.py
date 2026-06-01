"""
Microcap breakout backtest — does early-stage MICROCAP breakout momentum pay?
=============================================================================

STANDALONE RESEARCH TOOL (not wired into the engine, no endpoint, no DB writes,
no commit, no deploy). Per the owner's rule: *measure expectancy before shipping
a detector, cut losers.*

WHY THIS EXISTS
---------------
`engine/stage2_backtest.py` tested an RS + base-tightness filter on the engine's
LIQUID large/mega-cap UNIVERSE and found no edge. That tells us about big names.
The owner wants the REAL answer for actual MICROCAPS (low-priced, small,
sometimes-junky names), where breakout momentum lore is loudest — NOT a
large-cap proxy. So we build a genuine low-price / small universe from Alpaca's
live asset list and run a TIGHT-STOP, REALISTIC-COST trade simulation.

THE WHOLE POINT IS HONESTY ABOUT FRICTION + SURVIVORSHIP:
  * Microcaps gap THROUGH stops — we model fill-at-next-open when a bar gaps
    below the stop, not a fantasy fill at the stop price.
  * Microcap round-trip cost (spread + slippage) is large — we use 0.8% and
    report results BOTH with and without cost so the cost impact is explicit.
  * The universe is CURRENTLY-LISTED ONLY. Names that broke out and then died /
    delisted are GONE from Alpaca's active list, so they never enter the test.
    That is survivorship bias and it makes every number here OPTIMISTIC.

UNIVERSE CONSTRUCTION
---------------------
  1. alpaca-py TradingClient → all ACTIVE US_EQUITY assets, tradable=True.
  2. Batch a recent price for each (alpaca_client.get_latest_prices in chunks),
     keep names with last price in [PRICE_MIN, PRICE_MAX] (~$1-$15).
  3. Liquidity floor: ~20-day avg dollar volume from daily bars must be
     >= MIN_DOLLAR_VOL (default $1M/day) so they are at least tradeable.
  4. Cap to MAX_UNIVERSE (~200) via random sample (fixed seed) for sane runtime.

STRATEGY (per name, ~2y daily bars)
-----------------------------------
  * Entry event = today's close makes a FRESH new LOOKBACK_HIGH(=20)-day-high
    close (prior bar was NOT already extended) AND breakout-bar volume
    >= VOL_MULT(=2.0) x prior VOL_WINDOW(=20)-bar average volume.
  * Optional RS bucket: stock RS_WINDOW(=63)d return > SPY's over same window.

TRADE SIM (tight stop + realistic costs — the whole point)
----------------------------------------------------------
  * Entry price = breakout close.
  * Initial stop = max(entry*(1-MAX_STOP_PCT=0.08), breakout-bar low).
    (i.e. never risk more than 8%; if the bar's own low is tighter, use that.)
  * Walk forward bar-by-bar up to HORIZON(=40) trading days:
      - GAP-THROUGH: if the NEXT bar OPENS at/below the stop, fill at that OPEN
        (worse than the stop) and flag the trade as a gap-through.
      - else if the bar's intrabar LOW <= stop, fill at the stop.
      - else continue.
    If never stopped, exit at the close of the HORIZON-th bar.
  * Round-trip cost ROUNDTRIP_COST(=0.8%) is subtracted from the gross return.
    We report gross AND net so the friction is explicit.

METRICS (ALL breakouts, and the RS-filtered subset)
----------------------------------------------------
  N trades, win rate, mean/median return, mean win, mean loss, expectancy,
  and % of exits that were gap-throughs (filled worse than the stop).

USAGE
-----
    python -m engine.microcap_backtest
    python -m engine.microcap_backtest --max-universe 120 --min-dollar-vol 2e6
    python -m engine.microcap_backtest --price-min 1 --price-max 10 --days 760

Self-contained CLI. `import engine.config` loads the .env so the Alpaca clients
(data + trading) can auth.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Loads .env (Alpaca data + trading keys) as a side effect of import.
import engine.config  # noqa: F401
from engine.alpaca_client import get_bars, get_latest_prices

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("signalbolt.microcap_backtest")

# ── Universe-build tunables ───────────────────────────────────────────────────
PRICE_MIN        = 1.0       # keep names with a recent price >= this
PRICE_MAX        = 15.0      # ... and <= this (microcap-ish low-price band)
MIN_DOLLAR_VOL   = 1_000_000 # ~20-day avg dollar volume floor ($/day) — tradeable
MAX_UNIVERSE     = 200       # cap (random sample if more pass) to keep runtime sane
PRICE_CHUNK      = 200       # tickers per batched get_latest_prices call
RANDOM_SEED      = 13        # fixed seed so the random sample is reproducible

# ── Strategy tunables ─────────────────────────────────────────────────────────
LOOKBACK_HIGH    = 20        # "20-day high" breakout lookback (trading days)
VOL_WINDOW       = 20        # prior-bar average for volume confirmation
VOL_MULT         = 2.0       # breakout-bar volume >= this x prior 20-bar avg
RS_WINDOW        = 63        # relative-strength trailing window (~1 quarter)

# ── Trade-sim tunables ────────────────────────────────────────────────────────
MAX_STOP_PCT     = 0.08      # never risk more than 8% below entry
HORIZON          = 40        # max trading days to hold before exiting at close
ROUNDTRIP_COST   = 0.008     # 0.8% round-trip slippage+spread for microcaps

# Need LOOKBACK_HIGH + VOL_WINDOW behind the bar, RS_WINDOW behind it, and a full
# HORIZON of bars ahead. Pad so early-history edge cases are simply skipped.
MIN_BARS_REQUIRED = max(LOOKBACK_HIGH + 1, VOL_WINDOW, RS_WINDOW) + HORIZON + 5


@dataclass
class Trade:
    ticker:      str
    idx:         int     # integer position of the breakout bar
    rs_pass:     bool
    gross_ret:   float   # return BEFORE cost (fraction)
    net_ret:     float   # return AFTER ROUNDTRIP_COST (fraction)
    held_bars:   int     # trading days held until exit
    stopped:     bool    # True if exited on the stop (incl. gap-through)
    gap_through: bool    # True if filled WORSE than stop (next bar opened below)


# ── Data fetch ────────────────────────────────────────────────────────────────

def _fetch_daily(ticker: str, days: int, retries: int = 2) -> Optional[pd.DataFrame]:
    """Fetch daily bars, tolerating transient None responses + light rate-limit
    backoff. Returns a cleaned, time-sorted DataFrame or None if unusable."""
    for attempt in range(retries + 1):
        df = get_bars(ticker, timeframe="1Day", days=days)
        if df is not None and not df.empty:
            df = df.sort_index()
            df = df.dropna(subset=["open", "high", "low", "close"])
            return df if len(df) > 0 else None
        if attempt < retries:
            time.sleep(0.4)  # gentle backoff on transient None / rate limit
    return None


# ── Universe construction ──────────────────────────────────────────────────────

def _list_active_equities() -> list[str]:
    """All ACTIVE, tradable US_EQUITY symbols from Alpaca's TradingClient.
    Returns [] on any failure so the caller can bail gracefully."""
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus
    except Exception as e:
        logger.error("alpaca-py trading imports failed: %s", e)
        return []

    key    = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        logger.error("Alpaca API keys not set — cannot list assets.")
        return []

    try:
        tc = TradingClient(key, secret, paper=True)  # paper base is fine for asset listing
        req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        assets = tc.get_all_assets(req)
    except Exception as e:
        logger.error("get_all_assets failed: %s", e)
        return []

    syms = []
    for a in assets:
        try:
            if not a.tradable:
                continue
            sym = a.symbol
            # Skip exotic symbols (warrants/units/preferreds carry '.', '/', etc.)
            if not sym or any(c in sym for c in (".", "/", " ")):
                continue
            syms.append(sym)
        except Exception:
            continue
    return syms


def _batch_prices(symbols: list[str]) -> dict[str, float]:
    """Latest price for every symbol via chunked get_latest_prices calls."""
    out: dict[str, float] = {}
    for i in range(0, len(symbols), PRICE_CHUNK):
        chunk = symbols[i:i + PRICE_CHUNK]
        try:
            out.update(get_latest_prices(chunk))
        except Exception as e:
            logger.debug("price chunk %d failed: %s", i // PRICE_CHUNK, e)
        time.sleep(0.15)  # be gentle with the data API
    return out


def build_universe(days: int) -> list[str]:
    """Construct a low-price / small / liquidity-floored microcap-ish universe."""
    print("\nBuilding microcap universe from Alpaca's live asset list ...")
    all_syms = _list_active_equities()
    print(f"  Active tradable US equities listed: {len(all_syms)}")
    if not all_syms:
        return []

    # ── price filter ──
    prices = _batch_prices(all_syms)
    low_price = [s for s, p in prices.items() if PRICE_MIN <= p <= PRICE_MAX]
    print(f"  Priced names with last in ${PRICE_MIN:g}-${PRICE_MAX:g}: {len(low_price)}")
    if not low_price:
        return []

    # To bound the (expensive) per-name daily-bar liquidity check, pre-sample the
    # low-price pool generously (4x the cap) BEFORE the bar fetch, so we don't
    # pull bars for thousands of names. Random so it's not alphabetical-biased.
    random.seed(RANDOM_SEED)
    random.shuffle(low_price)
    liquidity_candidates = low_price[: MAX_UNIVERSE * 4]

    # ── liquidity floor: ~20-day avg dollar volume from daily bars ──
    print(f"  Checking ~{VOL_WINDOW}d avg $-volume >= ${MIN_DOLLAR_VOL:,.0f}/day "
          f"on {len(liquidity_candidates)} candidates ...")
    liquid: list[str] = []
    for k, sym in enumerate(liquidity_candidates, 1):
        if len(liquid) >= MAX_UNIVERSE:
            break
        try:
            df = _fetch_daily(sym, days=45)  # ~30 trading days is plenty for a 20d avg
            if df is None or len(df) < VOL_WINDOW:
                continue
            tail = df.tail(VOL_WINDOW)
            dollar_vol = float((tail["close"] * tail["volume"]).mean())
            if dollar_vol >= MIN_DOLLAR_VOL:
                liquid.append(sym)
        except Exception as e:
            logger.debug("liquidity check %s failed: %s", sym, e)
            continue
        if k % 100 == 0:
            print(f"    ... scanned {k}, kept {len(liquid)}")

    print(f"  Passed liquidity floor: {len(liquid)} (capped at {MAX_UNIVERSE})")
    return liquid[:MAX_UNIVERSE]


# ── SPY relative-strength benchmark ─────────────────────────────────────────────

def _build_spy_rs(days: int) -> Optional[pd.Series]:
    """SPY trailing RS_WINDOW return indexed by date (for the RS bucket)."""
    spy = _fetch_daily("SPY", days)
    if spy is None or len(spy) < RS_WINDOW + 5:
        return None
    c = spy["close"].to_numpy(dtype=float)
    vals = np.full(len(c), np.nan)
    for i in range(RS_WINDOW, len(c)):
        base = c[i - RS_WINDOW]
        if base > 0:
            vals[i] = (c[i] - base) / base
    return pd.Series(vals, index=spy.index).dropna()


# ── Per-ticker breakout scan + trade simulation ─────────────────────────────────

def _simulate_trade(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                    close: np.ndarray, i: int, n: int) -> Optional[tuple]:
    """Simulate one breakout trade entered at close[i]. Walk forward up to HORIZON
    bars with a tight stop, modelling gap-through-the-stop fills.

    Returns (gross_ret, held_bars, stopped, gap_through) or None if there isn't a
    full forward window (no look-ahead / no partial windows).
    """
    entry = close[i]
    if entry <= 0:
        return None
    if i + 1 >= n:  # need at least one bar after entry to manage the trade
        return None

    # Tight stop: never risk more than MAX_STOP_PCT; tighten to breakout-bar low.
    pct_stop = entry * (1.0 - MAX_STOP_PCT)
    stop = max(pct_stop, low[i])
    if stop >= entry:  # degenerate (low >= entry) — fall back to the pct stop
        stop = pct_stop

    last = min(i + HORIZON, n - 1)
    for j in range(i + 1, last + 1):
        # GAP-THROUGH: next bar opens at/below the stop → fill at the OPEN (worse).
        if open_[j] <= stop:
            gross = (open_[j] - entry) / entry
            return (gross, j - i, True, True)
        # Intrabar stop hit (no gap) → fill at the stop price.
        if low[j] <= stop:
            gross = (stop - entry) / entry
            return (gross, j - i, True, False)
    # Never stopped → exit at the close of the horizon bar.
    gross = (close[last] - entry) / entry
    return (gross, last - i, False, False)


def _scan_ticker(ticker: str, df: pd.DataFrame, spy_rs: Optional[pd.Series]) -> list[Trade]:
    """Emit a simulated Trade for every fresh volume-confirmed 20-day-high breakout."""
    trades: list[Trade] = []
    if df is None or len(df) < MIN_BARS_REQUIRED:
        return trades

    open_  = df["open"].to_numpy(dtype=float)
    high   = df["high"].to_numpy(dtype=float)
    low    = df["low"].to_numpy(dtype=float)
    close  = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    dates  = df.index
    n      = len(close)

    # Stock trailing RS_WINDOW return at each bar (for the RS bucket).
    stock_rs = np.full(n, np.nan)
    for i in range(RS_WINDOW, n):
        base = close[i - RS_WINDOW]
        if base > 0:
            stock_rs[i] = (close[i] - base) / base

    start = max(LOOKBACK_HIGH, VOL_WINDOW, RS_WINDOW)
    end   = n - 1  # need >=1 bar after entry; _simulate_trade enforces the rest
    for i in range(start, end):
        prior_high = np.max(close[i - LOOKBACK_HIGH:i])
        if not (close[i] > prior_high):
            continue
        # FRESH cross — previous bar must NOT already be above its own prior-20 high.
        prev_prior_high = np.max(close[i - 1 - LOOKBACK_HIGH:i - 1])
        if close[i - 1] > prev_prior_high:
            continue
        # Volume surge confirmation.
        prior_vol = volume[i - VOL_WINDOW:i]
        avg_vol = float(np.mean(prior_vol)) if len(prior_vol) else 0.0
        if not (avg_vol > 0 and volume[i] >= VOL_MULT * avg_vol):
            continue

        sim = _simulate_trade(open_, high, low, close, i, n)
        if sim is None:
            continue
        gross, held, stopped, gap = sim

        # RS bucket: stock 63d return > SPY 63d return at the breakout date.
        rs_pass = False
        s_rs = stock_rs[i]
        if spy_rs is not None and not np.isnan(s_rs):
            try:
                spy_at = spy_rs.asof(dates[i])
                if spy_at is not None and not (isinstance(spy_at, float) and np.isnan(spy_at)):
                    rs_pass = s_rs > float(spy_at)
            except Exception:
                rs_pass = False

        net = gross - ROUNDTRIP_COST
        trades.append(Trade(ticker, i, rs_pass, gross, net, held, stopped, gap))

    return trades


# ── Stats ────────────────────────────────────────────────────────────────────

def _bucket_stats(rets: list[float]) -> dict:
    """Headline stats for a list of trade returns (fractions)."""
    arr = np.array(rets, dtype=float)
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


def _fmt_pct(x) -> str:
    return "   n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+6.2f}%"


def _print_table(title: str, rows: list[tuple[str, dict]]) -> None:
    print(f"\n{title}")
    print("-" * 100)
    header = (f"{'bucket':<26}{'N':>6}{'win%':>8}{'mean':>9}"
              f"{'median':>9}{'mean_win':>10}{'mean_loss':>11}{'expectancy':>12}")
    print(header)
    print("-" * 100)
    for label, s in rows:
        if s["n"] == 0:
            print(f"{label:<26}{0:>6}{'   n/a':>8}{'   n/a':>9}{'   n/a':>9}"
                  f"{'   n/a':>10}{'   n/a':>11}{'   n/a':>12}")
            continue
        print(f"{label:<26}{s['n']:>6}{s['win_rate'] * 100:>7.1f}%"
              f"{_fmt_pct(s['mean']):>9}{_fmt_pct(s['median']):>9}"
              f"{_fmt_pct(s['mean_win']):>10}{_fmt_pct(s['mean_loss']):>11}"
              f"{_fmt_pct(s['expectancy']):>12}")
    print("-" * 100)


# ── Main ────────────────────────────────────────────────────────────────────

def run(days: int = 760) -> None:
    print("=" * 100)
    print("SignalBolt MICROCAP BREAKOUT BACKTEST — tight stop + realistic costs + gap-through risk")
    print("=" * 100)
    print(f"Price band: ${PRICE_MIN:g}-${PRICE_MAX:g}   |   Liquidity floor: "
          f"~{VOL_WINDOW}d avg $-vol >= ${MIN_DOLLAR_VOL:,.0f}/day   |   Universe cap: {MAX_UNIVERSE}")
    print(f"Breakout: fresh close > prior {LOOKBACK_HIGH}-bar high AND vol >= {VOL_MULT}x prior {VOL_WINDOW}-bar avg")
    print(f"Entry: breakout close   |   Stop: max(-{MAX_STOP_PCT*100:.0f}%, breakout-bar low)   |   "
          f"Horizon: {HORIZON} trading days")
    print(f"Gap-through: next bar opening <= stop fills at the OPEN (worse than stop)")
    print(f"Round-trip cost: {ROUNDTRIP_COST*100:.1f}% (reported gross AND net)   |   History: ~{days} calendar days")
    print("=" * 100)
    print("!! SURVIVORSHIP CAVEAT: universe is CURRENTLY-LISTED ONLY. Microcaps that broke out and then")
    print("!! died / delisted are absent from Alpaca's active list, so they NEVER enter this test. Every")
    print("!! number below is therefore OPTIMISTIC vs what a real-time trader would have experienced.")
    print("=" * 100)

    universe = build_universe(days)
    if not universe:
        print("\nFATAL: could not build a microcap universe (no names passed the filters). Aborting.")
        return
    print(f"\nFinal microcap universe: {len(universe)} names.")
    print("  " + ", ".join(universe[:40]) + (" ..." if len(universe) > 40 else ""))

    print("\nFetching SPY benchmark for the RS bucket ...")
    spy_rs = _build_spy_rs(days)
    print("  SPY RS series ready." if spy_rs is not None else "  SPY RS unavailable — RS bucket will be empty.")

    all_trades: list[Trade] = []
    scanned = skipped = 0
    print("\nScanning universe + simulating trades ...")
    for k, ticker in enumerate(universe, 1):
        try:
            df = _fetch_daily(ticker, days)
            if df is None or len(df) < MIN_BARS_REQUIRED:
                skipped += 1
                continue
            tr = _scan_ticker(ticker, df, spy_rs)
            all_trades.extend(tr)
            scanned += 1
            if k % 25 == 0 or k == len(universe):
                print(f"  [{k:>3}/{len(universe)}] scanned={scanned} skipped={skipped} "
                      f"trades so far={len(all_trades)}")
        except Exception as e:  # never let one bad ticker kill the run
            skipped += 1
            logger.warning("error scanning %s: %s", ticker, e)
            continue

    print(f"\nScanned {scanned} names, skipped {skipped}. Total simulated breakout trades: {len(all_trades)}")
    if not all_trades:
        print("No breakout trades found — nothing to measure.")
        return

    # ── Report: ALL breakouts and the RS-filtered subset, gross vs net ──
    def _report(label: str, trades: list[Trade]) -> None:
        if not trades:
            _print_table(f"{label} — (no trades)", [("GROSS", _bucket_stats([])),
                                                     ("NET (after cost)", _bucket_stats([]))])
            return
        gross = [t.gross_ret for t in trades]
        net   = [t.net_ret for t in trades]
        gap_n = sum(1 for t in trades if t.gap_through)
        stop_n = sum(1 for t in trades if t.stopped)
        avg_hold = float(np.mean([t.held_bars for t in trades]))
        _print_table(label, [
            ("GROSS (no cost)",       _bucket_stats(gross)),
            (f"NET (-{ROUNDTRIP_COST*100:.1f}% cost)", _bucket_stats(net)),
        ])
        print(f"  stopped out: {stop_n}/{len(trades)} ({stop_n/len(trades)*100:.1f}%)   |   "
              f"gap-through fills (worse than stop): {gap_n}/{len(trades)} "
              f"({gap_n/len(trades)*100:.1f}%)   |   avg hold: {avg_hold:.1f} bars")

    rs_trades = [t for t in all_trades if t.rs_pass]
    _report("ALL BREAKOUTS", all_trades)
    _report("RS-FILTERED (stock 63d > SPY 63d)", rs_trades)

    # ── Honest summary ──
    print("\n" + "=" * 100)
    print("HONEST SUMMARY")
    print("=" * 100)
    for label, trades in (("ALL breakouts", all_trades), ("RS-filtered", rs_trades)):
        if not trades:
            print(f"[{label}] no trades.")
            continue
        net = _bucket_stats([t.net_ret for t in trades])
        gross = _bucket_stats([t.gross_ret for t in trades])
        verdict = "POSITIVE" if net["expectancy"] > 0 else "NEGATIVE / no edge"
        print(f"[{label}] N={net['n']}  win%={net['win_rate']*100:.1f}  "
              f"gross expectancy={_fmt_pct(gross['expectancy'])}  "
              f"NET expectancy={_fmt_pct(net['expectancy'])}  -> {verdict}")
    print("-" * 100)
    print("Bottom line is the NET (after-cost) expectancy line above, and even that is OPTIMISTIC because")
    print("of survivorship (delisted blow-ups are excluded) and because overlapping same-ticker trades are")
    print("correlated, so treat N as smaller than it looks. Gap-through % shows how often the tight stop")
    print("did NOT save you (microcaps gapped past it). The 0.8% round-trip cost is conservative-to-light")
    print("for genuinely thin names — real fills are often worse. If NET expectancy is not solidly positive")
    print("AND robust across both buckets, early microcap breakout momentum is NOT a tradeable edge here.")
    print("=" * 100)


def main() -> None:
    global PRICE_MIN, PRICE_MAX, MIN_DOLLAR_VOL, MAX_UNIVERSE, VOL_MULT, ROUNDTRIP_COST, HORIZON, MAX_STOP_PCT
    ap = argparse.ArgumentParser(description="Microcap breakout backtest: tight stop + realistic costs + gap-through risk")
    ap.add_argument("--days", type=int, default=760, help="calendar days of daily history to fetch (~2y default)")
    ap.add_argument("--price-min", type=float, default=PRICE_MIN, help="min recent price for the universe")
    ap.add_argument("--price-max", type=float, default=PRICE_MAX, help="max recent price for the universe")
    ap.add_argument("--min-dollar-vol", type=float, default=MIN_DOLLAR_VOL, help="min ~20d avg $-volume/day floor")
    ap.add_argument("--max-universe", type=int, default=MAX_UNIVERSE, help="cap on number of names (random sample)")
    ap.add_argument("--vol-mult", type=float, default=VOL_MULT, help="breakout vol multiple vs prior 20-bar avg")
    ap.add_argument("--roundtrip-cost", type=float, default=ROUNDTRIP_COST, help="round-trip cost fraction (e.g. 0.008)")
    ap.add_argument("--horizon", type=int, default=HORIZON, help="max trading days to hold")
    ap.add_argument("--max-stop-pct", type=float, default=MAX_STOP_PCT, help="max stop distance below entry (fraction)")
    args = ap.parse_args()
    PRICE_MIN      = args.price_min
    PRICE_MAX      = args.price_max
    MIN_DOLLAR_VOL = args.min_dollar_vol
    MAX_UNIVERSE   = args.max_universe
    VOL_MULT       = args.vol_mult
    ROUNDTRIP_COST = args.roundtrip_cost
    HORIZON        = args.horizon
    MAX_STOP_PCT   = args.max_stop_pct
    run(days=args.days)


if __name__ == "__main__":
    main()
