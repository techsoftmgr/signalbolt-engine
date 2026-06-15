-- Market Pulse — one row per trading day. Run in the Supabase SQL editor.
-- Standalone market-wide regime read; separate from the signals tables.

create table if not exists market_pulse_daily (
  date date primary key,
  -- pillar 1: distribution days
  dd_count_spy        int     not null,
  dd_count_qqq        int     not null,
  -- pillar 2: new highs vs new lows (S&P 500)
  new_highs           int     not null,
  new_lows            int     not null,
  net_nhnl            int     not null,
  -- pillar 3: % above moving averages
  pct_above_50        numeric(5,2) not null,
  pct_above_200       numeric(5,2) not null,
  -- pillar 4: advance / decline
  ad_line_cumulative  bigint  not null,
  ad_advancers        int     not null,
  ad_decliners        int     not null,
  ad_divergence       boolean not null default false,
  -- pillar 5: VIX (nullable — secondary source may fail; never breaks the read)
  vix_close           numeric(6,2),
  vix_sma10           numeric(6,2),
  vix_band            text,            -- 'CALM' | 'NORMAL' | 'ELEVATED' | 'HIGH'
  vix_rising          boolean,
  -- verdict
  regime              text    not null check (regime in
                        ('CONFIRMED_UPTREND','UNDER_PRESSURE','CORRECTION')),
  guidance_key        text    not null,
  created_at          timestamptz not null default now()
);

-- Read path is public (free feature, served identically to everyone). Writes are
-- service-role only (the daily job). RLS: allow anon/auth SELECT, no client writes.
alter table market_pulse_daily enable row level security;

drop policy if exists market_pulse_read on market_pulse_daily;
create policy market_pulse_read on market_pulse_daily
  for select using (true);

-- (No insert/update/delete policy → only the service role key can write.)

create index if not exists market_pulse_daily_date_idx on market_pulse_daily (date desc);
