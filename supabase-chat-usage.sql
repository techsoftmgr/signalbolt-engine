-- AI stock-chat per-user daily quota.
-- Run once in the Supabase SQL editor. The engine (service role) reads/writes
-- this; the chat endpoint enforces the per-tier daily cap (see CHAT_DAILY_CAP
-- in main.py). One row per user per UTC day.

create table if not exists public.chat_usage (
    user_id    uuid        not null references auth.users(id) on delete cascade,
    day        date        not null,
    count      integer     not null default 0,
    updated_at timestamptz not null default now(),
    primary key (user_id, day)
);

-- Service-role only (the engine writes it); no end-user access needed.
alter table public.chat_usage enable row level security;

-- Optional cleanup: drop usage rows older than 30 days (keep the table small).
-- Run manually or via a scheduled job if desired:
--   delete from public.chat_usage where day < (now() at time zone 'utc')::date - 30;
