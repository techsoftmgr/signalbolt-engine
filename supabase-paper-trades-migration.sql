-- Paper-trading (admin-only) — proposals + paper-account execution lifecycle.
-- Idempotent: safe to re-run.

create table if not exists paper_trades (
  id              uuid primary key default gen_random_uuid(),
  signal_id       uuid,
  ticker          text not null,
  direction       text not null,                       -- LONG | SHORT
  qty             numeric not null,
  entry_price     numeric,
  stop_loss       numeric,
  target_one      numeric,
  alloc_usd       numeric,
  status          text not null default 'proposed',    -- proposed|submitted|filled|closed|rejected|canceled|error
  broker_order_id text,
  fill_price      numeric,
  exit_price      numeric,
  realized_pnl    numeric,
  realized_pct    numeric,
  strategy_type   text,
  detector_source text,
  note            text,
  created_at      timestamptz not null default now(),
  decided_at      timestamptz,
  closed_at       timestamptz
);

-- One proposal per signal (the propose scan relies on this to stay idempotent).
create unique index if not exists paper_trades_signal_uq on paper_trades (signal_id) where signal_id is not null;
create index if not exists paper_trades_status_idx on paper_trades (status);
create index if not exists paper_trades_created_idx on paper_trades (created_at desc);

-- Service-role only (admin reads/writes via the JWT-gated /admin/paper/* endpoints).
alter table paper_trades enable row level security;
-- (No public policies: RLS on + no policy = locked to the service role. Matches entry_gate_rejections.)
