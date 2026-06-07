-- Market Tape bias track record (V3).
-- One row per trading day: the day's risk-on/off bias + SPY price, then the
-- forward outcome (next-session SPY move) so we can show "bias was right X% of
-- the time". Service-role only (RLS on, no public policies) — written by the
-- engine, read via the admin endpoint.

create table if not exists market_bias_log (
  id                 uuid primary key default gen_random_uuid(),
  created_at         timestamptz not null default now(),
  bias               text,            -- 'risk-on' | 'risk-off' | 'neutral'
  vix                numeric,
  regime_type        text,
  spy_price          numeric,         -- SPY at snapshot time
  horizon_days       int,
  forward_price      numeric,         -- SPY at scoring time
  forward_return_pct numeric,
  correct            boolean,         -- did the bias match the forward move?
  scored_at          timestamptz
);

create index if not exists market_bias_log_created_idx on market_bias_log (created_at desc);

alter table market_bias_log enable row level security;
-- No public policies → only the service role (engine) can read/write.
