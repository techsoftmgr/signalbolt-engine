# Engine Backlog

Ideas captured for future implementation. Ordered by priority. Each item
includes scope, rationale, dependencies, and rough effort.

---

## 1. Sell Validator (exit-quality measurement) — ~1 day

**Priority:** HIGH (build after we have N≥50 closed signals from new pipeline)

**Problem:** Gate validator measures *"of signals we rejected, how many would have lost?"* (currently ~92%). We have no analog for closed signals: *"of signals we took, how much money did we leave on the table by exiting when we did?"*

**Design (mirrors `engine/gate_validator.py`):**

For each closed signal in the last 14 days, replay forward bars and compute:

- `max_favorable_after_exit` — best price reached AFTER our exit timestamp
- `pct_of_max_captured` — `realized_pnl / max_favorable_pnl` (1.0 = perfect exit, 0.5 = left half on table)
- `would_trailing_stop_have_done` — simulate a trailing-stop alternative exit
- `would_t3_have_hit` — would a 3× ATR target have been reached?
- `would_partial_exit_have_been_better` — simulate 50% at T1, 50% at T2

**Storage:** new columns on `signals` table OR a new `signal_exit_analysis` table.

**Headline metric:** average `pct_of_max_captured` across all closed signals.
- ≥ 0.7 → exits are reasonable
- 0.4-0.7 → leaving meaningful money on the table; tune exit logic
- < 0.4 → exit logic broken; tighten or trail more aggressively

**Endpoint:** `GET /admin/exit-validation-stats` returning headline + per-strategy + per-detector breakdown.

**App:** new "Exit Effectiveness" card on Admin → Analytics or a new screen.

---

## 2. Stop-Raid Post-Mortem & Recovery — ~1.5 days

**Priority:** MEDIUM (depends on Sell Validator first to quantify the problem)

**What exists today:**

| Layer | Mechanism |
|---|---|
| **Pre-fire detection** | `engine/manipulation_detector.py` watches recent bars for spike+reversal patterns. If detected, adds STOP_RAID flag, score -25 penalty |
| **SL placement** | `_round_number_adjustment` in `sl_tp_engine.py` nudges SL away from round numbers (where retail stops cluster) |
| **Gamma-aware SL** | SL placed BEYOND gamma support, not at obvious levels |
| **STOP_RAID widens SL** | When flag fires, SL widens further |

**What's missing:**

1. **Post-stop-out raid detection.** If our SL hits and price immediately reverses (>50% of stop distance recovered within 5 min), it was likely a raid. We don't flag this.

2. **Raid recovery / re-entry.** No mechanism to re-enter a position after a confirmed raid. Pros do this: raid stops them out → wait 1-5 min → re-enter when price reclaims the pre-raid level.

3. **Per-ticker raid history.** Some tickers get raided more often (low-float, illiquid, heavy short interest). We don't track or weight by historical raid rate.

**Design:**

- `engine/raid_detector.py` — runs in `signal_monitor` or as part of post-close analysis
- For each just-closed loss, fetch 5 min of forward bars
- If price reclaims `entry_price + 0.5 × (entry - stop)` within window → tag as `raided`
- Store in new `signal_postmortem` table column or `score_breakdown.raided = true`
- If `raid_recovery_enabled`, auto-fire a new signal on the same ticker with same direction once price reclaims (one re-entry max per session)

**Risk:** Re-entry can compound losses if the move was genuinely reversing, not a raid. Conservative threshold required.

---

## 3. T3 Target + Partial Exits — ~1 day

**Priority:** LOW (do after Sell Validator quantifies the value)

**Today:** signals exit 100% at T2. After T1, SL moves to breakeven, ride to T2.

**Proposal:** scale out instead of all-in/all-out:
- 33% close at T1 (lock in some profit)
- 33% close at T2 (current behavior)
- 33% trail with `entry + 0.5 × ATR(14)` stop, ride to T3 = 3× ATR target

**Why wait for Sell Validator:** if average `pct_of_max_captured` is already 0.8+, partial exits don't help. If it's 0.5, they could add 10-20% to average P/L.

**Risk:** more complexity in signal monitor + DB schema (needs `position_remaining_pct` column).

---

## 4. Trailing Stop After T1 — ~half day

**Priority:** LOW (subset of #3 — could ship this alone)

After T1 hits (currently moves SL to entry/breakeven), instead of holding SL fixed at entry, trail it with `current_price - 0.5 × ATR`. This protects profit if price continues up then reverses before hitting T2.

**Trade-off:** more frequent exits before T2, possibly lower per-trade max-P/L but higher average P/L because we capture more partial wins.

---

## 5. Signal Replayer (Chunk 3 from earlier roadmap) — ~2 days

**Priority:** MEDIUM (whenever we want to test a new gate or threshold)

**Purpose:** "what if?" tool for proposed code changes. Replays last N days of signals through new code to predict impact before deploying.

**Output:** table showing for each historical signal:
- Old code: fired Y/N, outcome
- New code: would have fired Y/N
- Net WR change

CLI: `python -m engine.signal_replayer --change=spread_threshold_0.2 --days=30`

**Why deferred:** speculative until we know what we want to tune. Sell validator + Chunks 1-2 of gate work tell us what to change first.

---

## 6. Quote subscription for proper buy/sell aggressor classification — ~1-2 hours

**Today:** block prints classified via tick rule (~70-75% accurate). Trade above prev = buy, below = sell, zero-tick = inherit.

**Upgrade:** subscribe to Alpaca quote stream (`StockDataStream.subscribe_quotes`), cache last bid/ask per ticker, classify trades via Lee-Ready (trade ≥ ask = buy, ≤ bid = sell, midpoint = unclear). ~85-90% accurate.

**Why deferred:** tick rule is honest enough for current use case. Upgrade only if classification accuracy becomes the bottleneck for a downstream feature (e.g. tape-direction gate).

---

## 7. Compression / Pullback predictive detectors — ✅ SHIPPED 2026-05-27

**Status:** Done. See `engine/compression_detector.py`, `engine/pullback_detector.py`.
After 1-2 weeks of fires, validate via:
```sql
SELECT score_breakdown->>'detector_source' AS source,
       COUNT(*), AVG(CASE WHEN result='win' THEN 1.0 ELSE 0 END) AS wr
FROM signals WHERE status='closed' GROUP BY 1;
```

---

## 8. Breakdown-quality study: volume + asset-class gate — ~half day (REVISIT ~late June 2026)

**Priority:** MEDIUM (needs ~3–4 weeks of post-2026-06-03 data first)

**Problem:** the BREAKDOWN detector (shorts a 20-day-low break) runs 80%+ in early
June 2026, but some breaks fail. Two suspected low-quality segments:

1. **Low volume (<1.5× rel volume)** — a break with no volume surge = no real
   selling pressure = likely false break. *Example:* GILD broke its 20-day low on
   only **1.2× volume** (2026-06-02) → reversed, exited −0.85% via structure-reversal.
2. **Commodity / macro ETFs** (GLD, SLV, GDX, USO, TLT…) mean-revert around macro
   levels far more than equities. *Example:* **GLD** broke its 20-day low (404.3) on
   a strong **3.0× volume** (2026-06-03), entry 406.97, stop 417.71 — yet it bounced
   to ~411 and the structure-reversal exit booked −0.40%. The breakdown simply
   doesn't follow through on gold the way it does on a momentum equity.

**Groundwork DONE (PR #201):** `breakdown_signals.py` now logs `relativeVolume`,
`asset_class` (commodity/bond/broad_etf/sector_etf/equity), and `is_etf` into each
SHORT card's `score_breakdown`. Logging-only — `classify_asset()` defaults unknown
tickers to `equity`, and tests prove entry/stop/targets are identical for equity vs
ETF (the live detector is untouched).

**The study:** once ~3–4 weeks of data accrue, segment realized expectancy:
```sql
SELECT score_breakdown->>'asset_class' AS asset_class,
       width_bucket((score_breakdown->>'relativeVolume')::float, 1.0, 4.0, 4) AS vol_bucket,
       COUNT(*) AS n,
       AVG(CASE WHEN result='win' THEN 1.0 ELSE 0 END) AS win_rate,
       AVG(result_pct) AS expectancy_pct
FROM signals
WHERE status='closed' AND score_breakdown->>'detector_source'='BREAKDOWN'
GROUP BY 1, 2 ORDER BY 1, 2;
```

**Decision rule (per no-proliferation):** if a cell is clearly negative with n≥~15,
add a gate in `breakdown_signals.generate` (e.g. require `relativeVolume >= 1.5`,
or skip `asset_class='commodity'`). DON'T tune on small samples — breakdown was only
~17 closed in 30 days. Historical expectancy is also noisy (phantom-price corruption,
fixed + corrected on 2026-06-03 — see `reference_signalbolt_phantom_price_guard`).

### Also measure: regime-conditioned expectancy (breakdown AND breakout)

Early (2026-06-03, ~3 days, single regime) BREAKDOWN ran 82% / +0.94%/trade while
the MIRROR BREAKOUT ran 35% / −0.40%/trade. Likely cause is **market regime, not
detector quality**: the tape was soft/choppy (SPY & IWM below their 10-day MA), which
rewards shorts (breakdown) and punishes longs (breakout — failed breakouts / bull
traps). The two also have different payoff shapes: breakdown = high win-rate, small
wins/losses (+1.3% / −0.8%); breakout = low win-rate (35%) but big wins/losses
(+3.5% / −2.5%) → a trend-following profile that needs a trending-UP market to pay.

So segment expectancy by `score_breakdown->>'regime_type'` (the engine already tags
it) for BOTH detectors:
```sql
SELECT score_breakdown->>'detector_source' AS det,
       regime_type, COUNT(*) AS n,
       AVG(CASE WHEN result='win' THEN 1.0 ELSE 0 END) AS win_rate,
       AVG(result_pct) AS expectancy_pct
FROM signals
WHERE status='closed' AND score_breakdown->>'detector_source' IN ('BREAKDOWN','BREAKOUT')
GROUP BY 1, 2 ORDER BY 1, 2;
```
**Likely fix (if confirmed):** gate **breakouts to bullish regimes** and **breakdowns
to bearish/weak regimes** — i.e. don't fire counter-trend. Should lift BOTH win rates.
Needs both an up AND a down regime in the sample before concluding; do NOT judge a
detector on a single-regime window. (Related: the deferred "extend market-regime
filter to SMC signals" backlog note.)

**3-state regime → detector map (theory to TEST per regime cell):**

| Regime | Favor (positive expectancy expected) | Suppress / down-weight |
|--------|--------------------------------------|------------------------|
| **Bull trend** | breakout (+ turnaround for pullback bottoms) | breakdown, peak |
| **Bear / weak** | breakdown (+ peak) | breakout, turnaround |
| **Chop / range** | mean-reversion: `vwap_reclaim`, `gap_fill`; fade-extremes: `turnaround` at strong support, `peak` at strong resistance; SMC liquidity-sweep reversals | **breakout & breakdown** (false breaks); also size DOWN — chop is low-edge for all |

Notes:
- Continuation detectors (breakout/breakdown) fail in chop (false breaks revert) —
  this is the leading explanation for breakout's early 35% / −0.40%/trade.
- Mean-reversion / fade-extremes carry **negative skew** in chop: many small wins
  until the range breaks into a trend → one big loss (short the top of what becomes
  a breakout). Measure expectancy AND tail, not just win-rate.
- Chop is low-edge for everyone; the right response may be "trade less / size down",
  not "switch detectors". Validate before gating.
- Verdict per cell needs adequate n IN THAT regime (≥~15). Don't conclude on a
  single-regime window. THEORY ONLY until the segmented `regime_type` data confirms it.

### Also: the OPTION (PUT/CALL) leg needs a higher bar + DTE/moneyness alignment

The breakdown stock short ran ~82% early, but the paired PUT did NOT (tiny sample:
JPM stopped out on a wiggle the short rode through; NFLX `expired` — right direction,
too slow for theta). This is **structural, not signal quality**: an option must
overcome theta + delta<1 + bid/ask spread + a tighter premium-stop, so the stock's
*small/slow* wins (many <1% structure-reversal exits) become option *losses*.

`result_pct`/`result_pnl` are NOW logged on option closes (premium P&L%), so option
expectancy is measurable. When data accrues, segment option expectancy by detector,
DTE bucket, and moneyness.

**UPDATE 2026-06-03 — SHIPPED:** `options_scanner` changed from 21–60 DTE / ~2% OTM
to **14–30 DTE / ~2% ITM** (delta ~0.6) for all option-firing detectors. The June
study should therefore compare option `result_pct` **before vs after this change**
(closed_at < 2026-06-03 = old params; >= = new) to confirm it actually helped.

**Original hypothesis (now live):** the old **21–60 DTE, ~2% OTM** picks were
misaligned with a 1–10 day swing.
- **Moneyness:** nearer-the-money / slightly-ITM (delta ~0.6) tracks the underlying
  more 1:1 and beats theta/spread better than 2% OTM (delta ~0.4).
- **DTE:** match the hold horizon + buffer. For a 10-trading-day swing that's roughly
  **~14–30 DTE** (NOT 60 — overpaying for time; NOT daily/0DTE — it expires before a
  multi-day swing resolves and has violent gamma/theta that worsens noise stop-outs).
  Daily/weekly expiries are for *intraday* theses, not multi-day swings.

---

## 9. Long-term TECHNICAL context for chat + Expert Read — ~half day

**Priority:** LOW-MED. Asked 2026-06-03: the AI chat can't answer "long-term"
questions because the entire grounding (quant verdict, Expert Read, signal levels)
is short-horizon (daily/1h/15m, 1-10 day swings). The assistant correctly refuses
to invent, so it stays silent on the long term.

**Scope (Option A — technical only, no fundamentals):** add a long-term block to
`chart_read.analyze` and the chat context built in `chat_service._build_context`:
- weekly/monthly bars → **200-day MA** (above/below), **52-week range + position
  within it**, multi-month trend & structure, distance from 52-wk high/low.
Cheap (just more Alpaca bars), stays in the technical lane (no investment-advice
liability), and lets chat answer "long-term trend / where it sits in its range /
is it above the 200-day". Keep declining true fundamentals (see #10 dependency).

## 10. Long-term "crash / deep-value" BUY signal (+ long-term SHORT) — LARGER, gated

**Priority:** MED (strategic differentiator) but **a different product track** — do
NOT bolt onto the swing detectors. Asked 2026-06-03: in a deep index drawdown
(-20-30% from highs) quality names trade at multi-year lows; a long-term buy at the
lows historically returns 100-200% on recovery (2008/2020/2022).

**Why it's a separate track, not a quick detector:**
- **Horizon:** months-to-years. The whole engine assumes 1-10 day swings (tight
  stops, 10-day expiry, 5-min monitors). A long-term position would get shaken out
  by those. Needs the existing `position_trade` type (720h) extended with LOOSE
  management: wide/no hard stop, scale-in, hold through volatility.
- **Fires RARELY:** regime-gated (once every few years), so it's an occasional
  "generational buy" alert, not a steady signal stream. Different UX + audience
  (investors, not traders).
- **DEPENDS ON FUNDAMENTALS (the blocker):** "good stocks trading deep low" can't be
  picked on technicals alone — you'd catch falling knives / value traps (a stock
  -30% can go -60%; survivorship bias hides the bankruptcies). Needs a quality screen
  (balance sheet, earnings durability) = the fundamentals feed we don't have yet
  (Option B from the long-term-context discussion).

**Buildable now (cheap, technical, no fundamentals):** a **market-regime drawdown
detector** — index (SPY/QQQ/IWM) % off its 52-week high + breadth — that flags
"deep drawdown regime." That's the trigger half. Layer the quality screen + the
long-term BUY list once fundamentals exist.

**UPDATE 2026-06-04 — BOTH HALVES SHIPPED (signal not yet fired):**
- ✅ **Quality half** — `engine/fundamentals.py` (PR #215/#216): robust EDGAR XBRL
  extraction (largest-annual-per-FY revenue + sanity guards — fixed the META 148%/
  MSFT 163% bug), net margin / ROE / D/E / growth / FCF → 0-5 quality_score. Full
  **S&P 500 universe** (live CSV w/ CIKs, cached daily), `fundamentals_cache` table
  + rolling refresh (3×/day, batch 20). `GET /admin/quality-screen`.
- ✅ **Regime half (Phase 0)** — `engine/drawdown_regime.py` (PR #217): SPY/QQQ/IWM
  % off 52-wk high → healthy/pullback/correction/bear/deep_bear + `accumulation_window`
  (opens at SPY <= -20%). `GET /market/regime-drawdown`. Daily ops log 4:12 PM ET.
- ⬜ **REMAINING = combine + surface (the actual signal — NOT built):** when
  `accumulation_window` is open, take quality_score>=4 from fundamentals_cache → a
  long-horizon BUY list using `position_trade` (720h) with LOOSE mgmt (wide/no hard
  stop, scale-in, hold through vol — do NOT reuse the swing monitors). Surface via a
  regime BANNER on Signals/Markets + Position-tagged picks in the Signals tab (NO new
  tab — bar is full at 6) + optional push when regime enters bear. This is the bigger
  product + investment-advice-framing decision — paused pending user go-ahead.

**Long-term SHORT (inverse):** overvalued names at market tops. Lower priority +
harder — shorting has unlimited risk, borrow cost, and markets stay irrational longer
than a short can hold. Defer.

**Liability:** long-term buy/sell recommendations are heavier investment-advice
territory than a 1-10 day signal — keep the "educational, not advice" framing.

---

## How to use this backlog

- Items get re-prioritised whenever new data changes the calculus
- Each item must have **measurable rationale** before promotion to active work
- Don't build #2-#4 until #1 (Sell Validator) tells us if exit quality is actually the leak
- Don't build #6 until tape direction becomes a real bottleneck
