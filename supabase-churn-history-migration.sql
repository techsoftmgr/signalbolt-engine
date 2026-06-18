-- Churn/Absorption occurrence log → resolution scorecard (measure-first, no firing change).
-- One row per (trading session × ticker) that showed up on the Churn/Absorption screen, plus the
-- multi-day "coiling" streak. Forward outcomes are computed ON THE FLY from daily bars (like
-- breakout_watch_history) — this table only persists the OCCURRENCE so we can score it later.
create table if not exists churn_history (
  id            uuid primary key default gen_random_uuid(),
  session_date  date not null,
  ticker        text not null,
  zone          text not null,            -- accumulation (near lows) | distribution (near highs) | churn (mid)
  rel_vol       numeric,                  -- pace-adjusted relative volume at observation
  churn_score   numeric,                  -- volume per unit of price move (absorption strength)
  range_pos     numeric,                  -- 0..1 position in the 20-day range
  event         boolean default false,    -- recent news-gap pin (M&A) → technical read may not apply
  obs_close     numeric,                  -- close at observation = entry for the forward-return calc
  streak        int default 1,            -- consecutive trading sessions this ticker has been absorbing (coiling)
  created_at    timestamptz default now(),
  unique (session_date, ticker)
);

create index if not exists idx_churn_history_date   on churn_history (session_date);
create index if not exists idx_churn_history_ticker on churn_history (ticker);

-- Market-data, low sensitivity: public read; writes via service role only (no insert/update policy).
alter table churn_history enable row level security;
do $$ begin
  create policy churn_history_read on churn_history for select using (true);
exception when duplicate_object then null; end $$;
