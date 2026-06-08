-- Read self-grade (Pass 2). One row/day per ticker capturing the levels the
-- daily Expert Read flagged, then the forward outcome: did those levels behave
-- as support/resistance? Descriptive accuracy of the read — NOT a win-rate.
-- Service-role only (RLS on, no public policies). Written by the engine.

create table if not exists read_accuracy_log (
  id                uuid primary key default gen_random_uuid(),
  created_at        timestamptz not null default now(),
  ticker            text not null,
  bias              text,
  price             numeric,
  support           numeric,
  resistance        numeric,
  bull_trigger      numeric,
  bear_trigger      numeric,
  horizon_days      int,
  support_tested    boolean,
  support_held      boolean,
  resistance_tested boolean,
  resistance_held   boolean,
  scored_at         timestamptz
);

create index if not exists read_accuracy_log_ticker_idx on read_accuracy_log (ticker, created_at desc);
create index if not exists read_accuracy_log_unscored_idx on read_accuracy_log (scored_at) where scored_at is null;

alter table read_accuracy_log enable row level security;
-- No public policies → only the service role (engine) can read/write.
