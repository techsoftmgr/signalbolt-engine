# Market Pulse

A standalone, **market-wide**, end-of-day **regime read** (IBD-style). Computed once
daily after the close, written to one Supabase row per day, served identically to
everyone. **Completely separate from the per-signal confluence engine** — it touches
no signal logic.

It answers: *what regime is the broad market in, and what does that regime
historically imply?* — in plain language, with no numeric score shown.

## The five pillars

| # | Pillar | Measures | Source |
|---|--------|----------|--------|
| 1 | **Distribution-Day Count** | institutional selling pressure | Alpaca — SPY + QQQ |
| 2 | **Net New Highs vs New Lows** | breadth extremes | Alpaca — S&P 500 |
| 3 | **% Above 50d / 200d MA** | breadth participation | Alpaca — S&P 500 |
| 4 | **Advance/Decline Line** | breadth divergence | Alpaca — S&P 500 |
| 5 | **VIX Level & Trend** | expected volatility / fear | **secondary source (NOT Alpaca)** |

- **Pillar 1** — index closes down ≥ 0.2% on higher volume than the prior day; rolling
  25 trading days; a day expires after 25 days **or** once the index closes 5%+ above
  it (both rules implemented). `dd_max = max(SPY, QQQ)`.
- **Pillar 2** — per S&P 500 name, today’s high exceeds all prior highs in 252 days
  (new high) / low below all prior lows (new low). `net = highs − lows`.
- **Pillar 3** — % of names above their 50-day / 200-day SMA. Healthy > 50%, weak < 40%.
- **Pillar 4** — daily advancers − decliners → a **cumulative** A/D line. Divergence =
  SPY within 0.5% of its 52-week high while the A/D line is **not** at a new high.
- **Pillar 5** — VIX close vs its own 10-day SMA (rising/falling) + band
  (CALM < 15, NORMAL 15–20, ELEVATED 20–30, HIGH > 30).

## Regime tiers (top-down, first match wins; no score shown)

```
CORRECTION      if  dd_max>=6  OR  %above200<40  OR  (net_nhnl<0 AND %above50<40)
UNDER_PRESSURE  if  dd_max>=5  OR  ad_divergence  OR  %above50<50  OR  net_nhnl<0
                    OR (vix>30 AND vix_rising)            # VIX soft confirmer
                # boundary: otherwise-CONFIRMED but dd_max==4 AND vix>25 AND rising → UNDER_PRESSURE
else CONFIRMED_UPTREND
```

**VIX is a soft confirmer only** — it can nudge a borderline read but never, by itself,
create a CORRECTION. All thresholds are named constants in `config.py`.

## VIX secondary source + isolation (why)

VIX is **not on Alpaca** (Alpaca carries US stocks/ETFs only — "asset not found for
VIX"). `data.vix_closes()` fetches it from **yfinance `^VIX`** (Cboe-derived **Stooq**
CSV fallback). The fetch is fully isolated: on any failure it returns `None`, the row
stores `vix_*` as null, the UI shows "VIX unavailable", and the **regime is computed
from pillars 1–4 only**. VIX can never be a single point of failure.

## Endpoints

- `GET /market-pulse/today` — latest row + full guidance text + VIX line + disclaimer (public).
- `GET /market-pulse/history?days=90` — arrays for charting (net_nhnl, %above 50/200,
  cumulative A/D, dd counts, VIX, regime) (public).
- `POST /admin/run-market-pulse` — compute today now (admin).
- `POST /admin/run-market-pulse?backfill_days=120` — seed the A/D line + history (admin).

## Schedule

Worker cron at **4:45 PM ET** (DST-aware, `America/New_York`), Mon–Fri, trading-day
gated. Idempotent upsert on `date` (re-runs overwrite; A/D uses the **prior** day's
cumulative so it never double-counts).

## First-time setup

1. Run `supabase-market-pulse-migration.sql` in the Supabase SQL editor.
2. Seed history + the A/D line: `POST /admin/run-market-pulse?backfill_days=120`
   (one-time; replays the last ~120 trading days from one bulk fetch).
3. After that the daily cron keeps it current.

## Phase 2 additions

### A — Stalling days (Pillar 1 extension)
A *stalling* day is softer distribution: close **UP** but a tiny gain
(≤ `STALL_MAX_GAIN_PCT`, 0.2%), on **higher** volume, closing in the **lower half**
of the range (`(close-low)/(high-low) ≤ STALL_CLOSE_RANGE_FRAC`) — institutions
selling into strength. Same 25-day window + 5%-rise expiry as distribution days.
Combined pressure: `effective_dd = dd_count + STALL_WEIGHT*stall_count` (0.5 → two
stalls ≈ one distribution). The regime resolver compares against
`floor(effective_dd_max)` instead of `dd_max` (thresholds 5/6 unchanged) — with
zero stalling days, identical to Phase 1. Stored/served separately
(`stall_count_*`, `effective_dd_*`) so the UI shows "Distribution: X · Stalling: Y".
Migration: `supabase-market-pulse-stalling-migration.sql`.

### B — Intraday provisional read (`intraday.py`)
`GET /market-pulse/intraday` — is today *on pace* for a distribution/stalling day,
**before** the close. **INTEGRITY FIREWALL:** the module has no DB-write path (no
`store` import, no `upsert`, no `.table()`) — it can never reach `market_pulse_daily`.
Volume is projected with an **empirical U-curve** (trailing ~60 sessions of 30-min
buckets, cached; hardcoded U-curve fallback) — NOT linear off the clock, which
understates midday. **Confidence gating:** `TOO_EARLY` before 11:00 ET, then
MEDIUM → HIGH; status only flips when projected volume clears prior-day volume by
`INTRADAY_MARGIN`. Half-days use the real NYSE session length + a separate curve.
Every field is flagged provisional; out-of-hours → `MARKET_CLOSED`.

## TODOs

- **Quarterly:** constituents come from `fundamentals.get_universe()` (a maintained
  CSV, cached). If S&P 500 coverage drifts, force a refresh there.
- A/D **divergence** needs accrued A/D history; it stays `false` during the initial
  backfill window and becomes meaningful as days accumulate.
- Intraday: swap the hardcoded U-curve for the empirical one (already wired — it's
  built lazily + cached 24h); a separate **empirical half-day curve** is a TODO
  (currently a hardcoded half-day fallback).
- Sector Leaders (Part C) lives in its own package `engine/sector_leaders/` — see
  `GET /sector-leaders/today` + `/history`; backfill via
  `POST /admin/run-sector-leaders?backfill_days=130`. Migration:
  `supabase-sector-leaders-migration.sql`.
