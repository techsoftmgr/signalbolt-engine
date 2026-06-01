-- ─────────────────────────────────────────────────────────────────────────────
-- In-app Alerts feed (shared, event-level).
--
-- Powers the new "Alerts" tab so users see EVERY engine alert (new signals,
-- watchlist state-changes, breakdown early/confirmed, cycle, buzz, stop-raised)
-- IN THE APP — even when OS push isn't delivering (e.g. FCM not yet configured
-- on a standalone Android build). The engine inserts one row per alert event via
-- engine/push.py::_record_alert(); the app reads + subscribes via Realtime.
--
-- Run once in the Supabase SQL editor.
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.alerts (
  id         uuid primary key default gen_random_uuid(),
  type       text not null,                       -- signal | watchlist | breakdown | cycle | buzz | stop
  ticker     text,
  stage      text,                                -- early | confirmed | buyzone | topping | peak | ...
  title      text not null,
  body       text,
  data       jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists alerts_created_at_idx on public.alerts (created_at desc);
create index if not exists alerts_type_idx       on public.alerts (type);
create index if not exists alerts_ticker_idx     on public.alerts (ticker);

alter table public.alerts enable row level security;

-- Any authenticated user can READ the shared feed.
drop policy if exists "alerts_read_authenticated" on public.alerts;
create policy "alerts_read_authenticated"
  on public.alerts for select
  to authenticated
  using (true);

-- INSERTs come only from the engine (service_role bypasses RLS), so we grant no
-- insert/update/delete policy to anon/authenticated.

-- Realtime so the Alerts tab shows new rows live + drives the unread badge.
-- (Wrapped so re-running this file doesn't error if it's already in the pub.)
do $$
begin
  alter publication supabase_realtime add table public.alerts;
exception
  when duplicate_object then null;
  when others then null;
end $$;

-- Optional retention: keep the feed lean. Uncomment to drop alerts older than 30d.
-- delete from public.alerts where created_at < now() - interval '30 days';
