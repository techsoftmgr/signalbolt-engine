-- ─────────────────────────────────────────────────────────────────────────────
-- Breakout Watch lifecycle history.
-- ONE row per WATCH EPISODE (a ticker's continuous stay on the breakout watch),
-- NOT one per 60s dashboard refresh. The engine job (breakout_watch.sync_watch)
-- opens an episode when a ticker enters the breakout bucket, updates it each
-- cycle (peak / max move / trigger), and closes it on exit.
--
-- States:  WATCHING -> TRIGGERED  (price broke above the 20-day high)
--                   -> FADED      (left the breakout zone without breaking)
--                   -> EXPIRED    (stale on watch with no breakout)
-- outcome (win|loss) is backfilled later by breakout_validator → Watch Accuracy.
--
-- Run this in the Supabase SQL editor. Standard PostgreSQL only.
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists breakout_watch_history (
    id                 uuid primary key default gen_random_uuid(),
    ticker             text not null,
    session_date       date not null default (now() at time zone 'utc')::date,
    state              text not null default 'WATCHING',   -- WATCHING|TRIGGERED|FADED|EXPIRED
    entered_at         timestamptz not null default now(),
    enter_price        numeric,
    breakout_level     numeric,        -- 20-day high at entry (the level being tested)
    enter_score        numeric,        -- breakout/quant score at entry
    last_seen_at       timestamptz not null default now(),
    triggered_at       timestamptz,
    trigger_price      numeric,
    exited_at          timestamptz,
    exit_reason        text,           -- TRIGGERED|FADED|EXPIRED
    peak_price         numeric,        -- best price seen during the episode
    max_favorable_pct  numeric,        -- MFE % from enter_price
    outcome            text,           -- win|loss|null (unjudged)
    realized_pct       numeric,
    updated_at         timestamptz not null default now()
);

-- At most ONE open episode per ticker (so the 60s/5-min cycle updates, not duplicates).
create unique index if not exists ux_breakout_watch_open
    on breakout_watch_history (ticker)
    where exited_at is null;

create index if not exists ix_breakout_watch_session on breakout_watch_history (session_date);
create index if not exists ix_breakout_watch_outcome on breakout_watch_history (outcome) where outcome is not null;

-- Internal / service-role only: the engine writes via the service key (which
-- bypasses RLS), and the app reads through /quant/dashboard server-side. Enabling
-- RLS with no policy blocks any direct anon/authenticated access.
alter table breakout_watch_history enable row level security;
