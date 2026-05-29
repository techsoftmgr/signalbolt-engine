-- ─────────────────────────────────────────────────────────────────────────────
-- Generalize breakout_watch_history into a multi-bucket "setup watch" ledger.
--
-- Adds a `bucket` column so the SAME table + sync + validator + history/scorecard
-- machinery tracks EVERY Quant dashboard section (breakout, breakdown,
-- momentum, pullback, high-volume, vwap-reclaim, oversold-bounce) — one row per
-- watch EPISODE per bucket, not per refresh.
--
-- Non-destructive: existing rows default to bucket='breakout'. The one-open-
-- episode-per-ticker guard is relaxed to one-open per (ticker, bucket) so the
-- same name can be live in more than one bucket at once.
--
-- Run this in the Supabase SQL editor. Standard PostgreSQL only.
-- ─────────────────────────────────────────────────────────────────────────────

alter table breakout_watch_history
    add column if not exists bucket text not null default 'breakout';

-- Relax the unique "one open episode per ticker" index to (ticker, bucket).
drop index if exists ux_breakout_watch_open;
create unique index if not exists ux_breakout_watch_open
    on breakout_watch_history (ticker, bucket)
    where exited_at is null;

create index if not exists ix_breakout_watch_bucket on breakout_watch_history (bucket);
