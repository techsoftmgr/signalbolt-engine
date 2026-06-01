-- ============================================================================
-- Community Tab — social trending time-series (snapshot history)
-- ============================================================================
-- Stores periodic snapshots of the merged trending feed (Reddit/Apewisdom +
-- StockTwits) together with the price at capture time. This is the foundation
-- that turns a live "what's loud" list into measurable insight:
--
--   • Going-viral detection  — mention z-score vs each ticker's OWN baseline
--   • Buzz velocity          — rate-of-change of mentions across recent snaps
--   • 7-day sparklines        — attention building vs fading
--   • "What changed"          — new entrants / climbers / fallers / first-time
--   • Trending → returns      — forward return vs SPY from the price at capture
--
-- Written hourly by the engine scheduler (runner.py :: _run_social_snapshot)
-- using the service-role client. Reads happen server-side through the engine,
-- so RLS stays locked (no public policies = service role only).
-- ============================================================================

create table if not exists social_snapshots (
    id                  bigserial primary key,
    captured_at         timestamptz not null default now(),
    ticker              text        not null,
    name                text,
    rank                int,            -- overall rank in our merged trending list (1 = hottest)
    score               numeric,        -- combined trending score (drives sort)
    reddit_mentions     int,
    reddit_rank         int,
    reddit_sentiment    numeric,        -- 0..1 bullish weight
    stocktwits_rank     int,
    stocktwits_watchers int,
    sources             text[],
    price               numeric         -- price at capture time (NULL if unavailable)
);

-- Per-ticker history lookups (baseline, velocity, sparkline, forward returns)
create index if not exists idx_social_snap_ticker_time
    on social_snapshots (ticker, captured_at desc);

-- Time-window scans ("what changed", track-record windows, retention pruning)
create index if not exists idx_social_snap_time
    on social_snapshots (captured_at desc);

alter table social_snapshots enable row level security;

-- ── Optional housekeeping ───────────────────────────────────────────────────
-- Snapshots accrue ~50 rows/hour. Keep ~120 days for the track record, prune
-- the rest. Run manually or wire to pg_cron if available:
--   delete from social_snapshots where captured_at < now() - interval '120 days';

comment on table social_snapshots is
    'Hourly snapshots of the Community trending feed + price at capture. '
    'Powers going-viral z-scores, buzz velocity, sparklines, what-changed, '
    'and the trending->returns track record. Service-role writes only.';
