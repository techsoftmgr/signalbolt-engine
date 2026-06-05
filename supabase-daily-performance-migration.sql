-- Daily EOD performance snapshot — ONE immutable row per trading day.
-- The synthesis layer over the per-signal MFE capture + the regime timeline:
-- records that day's CLOSED outcomes (by detector / conviction / direction),
-- profit give-back (peak vs realized), the day's regime path, and the ACTIVE
-- book state. A longitudinal series that powers the deferred studies (giveback
-- backtest, regime-conditional entry, conviction calibration) + early detection.
-- Written ~8:05 PM ET, after the full extended session (4 AM–8 PM MFE window).
--
-- Run once in the Supabase SQL editor.

create table if not exists daily_performance (
  id                    uuid primary key default gen_random_uuid(),
  trade_date            date not null unique,
  created_at            timestamptz not null default now(),

  -- market context (from regime_history)
  regime_close          text,        -- regime at the close
  regime_path           text,        -- the day's transitions, e.g. "pre TRENDING_BULL > rth PANIC"
  vix                   numeric,

  -- CLOSED today (realized, direction-aware result_pct)
  closed_n              integer,
  closed_wins           integer,
  closed_win_rate       numeric,
  closed_net_pct        numeric,
  closed_avg_pct        numeric,
  closed_profit_factor  numeric,
  carried_n             integer,     -- of the closed, how many were opened on a prior day
  long_n                integer,
  long_win_rate         numeric,
  long_net_pct          numeric,
  short_n               integer,
  short_win_rate        numeric,
  short_net_pct         numeric,
  giveback_pct          numeric,     -- sum of (peak MFE − realized) over closed-today

  by_detector           jsonb,       -- {detector: {n, wins, net, avg}}
  by_conviction         jsonb,       -- {tier: {n, wins, net, avg}}
  top_winner            jsonb,       -- {ticker, detector, direction, pct}
  top_loser             jsonb,

  -- ACTIVE book snapshot (8 PM ET; AH marks are thinner — supplementary)
  active_n              integer,
  active_net_unreal_pct numeric,
  active_long_n         integer,
  active_short_n        integer,
  active_near_levels    integer,     -- # active within ~1.5% of stop or target
  active_giveback_pct   numeric      -- sum of (peak MFE − current unrealized) over active
);

create index if not exists idx_daily_perf_date on daily_performance (trade_date desc);

alter table daily_performance enable row level security;
