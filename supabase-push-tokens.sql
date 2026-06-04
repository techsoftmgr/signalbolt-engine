-- ─────────────────────────────────────────────────────────────────────────────
-- Multi-device push tokens.
--
-- Until now a user had ONE push_token on their profile, so logging in on a 2nd
-- device (or re-installing) OVERWROTE the first — only the latest device got
-- pushes. This table lets one user register MANY devices; the engine sends to
-- the UNION of a user's tokens (engine/push.py::_device_rows + register_device).
--
-- A given Expo token belongs to exactly one install, so `token` is UNIQUE and
-- gets REASSIGNED to the new user on re-login (handled by register_device).
--
-- Run once in the Supabase SQL editor. Safe to re-run.
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.push_tokens (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null references public.profiles(id) on delete cascade,
  token      text not null unique,            -- ExponentPushToken[...]
  platform   text,                            -- 'android' | 'ios'
  created_at timestamptz not null default now(),
  last_seen  timestamptz not null default now()
);

create index if not exists push_tokens_user_id_idx on public.push_tokens (user_id);

-- Registration goes through the engine (service_role bypasses RLS), so we keep
-- RLS on and grant users read of their OWN devices only. No client writes.
alter table public.push_tokens enable row level security;

drop policy if exists "push_tokens_read_own" on public.push_tokens;
create policy "push_tokens_read_own"
  on public.push_tokens for select
  to authenticated
  using (user_id = auth.uid());

-- One-time backfill: seed the table from the existing single-token column so no
-- one loses pushes during the migration. ON CONFLICT keeps the table idempotent.
insert into public.push_tokens (user_id, token, last_seen)
select id, push_token, now()
from public.profiles
where push_token is not null
  and push_token like 'ExponentPushToken[%'
on conflict (token) do nothing;
