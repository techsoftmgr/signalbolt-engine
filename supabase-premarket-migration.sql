-- ============================================================
-- SignalBolt — Pre-market Watchlist Migration
-- Run this in: Supabase Dashboard → SQL Editor
-- Safe to re-run (all statements are idempotent)
-- ============================================================

-- ── 1. Create premarket_watchlist table ──────────────────────
CREATE TABLE IF NOT EXISTS premarket_watchlist (
    id                 BIGSERIAL PRIMARY KEY,
    ticker             TEXT        NOT NULL,
    scan_date          DATE        NOT NULL,
    pm_gap_pct         NUMERIC(8,4) DEFAULT 0,
    pm_direction       TEXT        DEFAULT 'FLAT',     -- UP | DOWN | FLAT
    pm_high            NUMERIC(12,4) DEFAULT 0,
    pm_low             NUMERIC(12,4) DEFAULT 0,
    pm_latest_price    NUMERIC(12,4) DEFAULT 0,
    pm_volume          BIGINT       DEFAULT 0,
    pm_volume_ratio    NUMERIC(8,3) DEFAULT 0,
    pm_has_news        BOOLEAN      DEFAULT FALSE,
    pm_news_headline   TEXT         DEFAULT '',
    watch_score        INTEGER      DEFAULT 0,         -- 0-100
    watch_reasons      TEXT[]       DEFAULT '{}',
    prior_close        NUMERIC(12,4) DEFAULT 0,
    scanned_at         TIMESTAMPTZ  DEFAULT NOW(),
    created_at         TIMESTAMPTZ  DEFAULT NOW(),

    -- Each ticker appears once per scan_date; 9 AM scan overwrites 8 AM
    UNIQUE (ticker, scan_date)
);

-- ── 2. Indexes ────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_premarket_scan_date
    ON premarket_watchlist (scan_date DESC);

CREATE INDEX IF NOT EXISTS idx_premarket_watch_score
    ON premarket_watchlist (scan_date DESC, watch_score DESC);

CREATE INDEX IF NOT EXISTS idx_premarket_ticker
    ON premarket_watchlist (ticker, scan_date DESC);

-- ── 3. Row-Level Security ─────────────────────────────────────
ALTER TABLE premarket_watchlist ENABLE ROW LEVEL SECURITY;

-- Allow all authenticated users to read
DROP POLICY IF EXISTS "premarket_read" ON premarket_watchlist;
CREATE POLICY "premarket_read"
    ON premarket_watchlist
    FOR SELECT
    TO authenticated
    USING (true);

-- Only the service role (engine) can write
DROP POLICY IF EXISTS "premarket_write" ON premarket_watchlist;
CREATE POLICY "premarket_write"
    ON premarket_watchlist
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- ── 4. Keep only 7 days of history (run manually or via pg_cron) ──
-- SELECT cron.schedule('premarket-cleanup', '0 10 * * *',
--   'DELETE FROM premarket_watchlist WHERE scan_date < NOW() - INTERVAL ''7 days''');
-- Uncomment the above if you have pg_cron enabled in Supabase.

-- ── 5. Quick verification ─────────────────────────────────────
SELECT
    'premarket_watchlist created' AS status,
    COUNT(*) AS row_count
FROM premarket_watchlist;
