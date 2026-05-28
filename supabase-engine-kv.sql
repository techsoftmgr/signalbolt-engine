-- engine_kv: durable key/value store for engine runtime snapshots.
-- Replaces the flaky Redis snapshot for armed per-tick zones (stream:zones:v1)
-- so accumulated compression/pullback/swing zones survive worker restarts and
-- the admin display reads a reliable source (no more Redis socket timeouts).
--
-- Run once in the Supabase SQL editor.

create table if not exists engine_kv (
  key        text primary key,
  value      jsonb       not null,
  updated_at timestamptz not null default now()
);

-- Service-role only. RLS on with no policies = the engine's service key (which
-- bypasses RLS) is the only thing that can read/write. No anon/user access.
alter table engine_kv enable row level security;
