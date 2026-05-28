-- ============================================================
-- Armed-zone lifecycle history
-- Captures every predictive zone arming + its outcome (fired /
-- expired) so we can analyse conversion rate and win-rate per
-- detector over time. Replaces the old behaviour of trashing
-- zones at the overnight clear with no trace.
--
-- Run in: Supabase Dashboard → SQL Editor (safe to re-run).
-- Standard PostgreSQL only.
-- ============================================================

CREATE TABLE IF NOT EXISTS armed_zone_history (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker          TEXT         NOT NULL,
    detector        TEXT         NOT NULL,   -- COMPRESSION / PULLBACK / SWING_BREAKOUT
    direction       TEXT,                    -- LONG / SHORT (null at arm for compression)
    armed_level     DOUBLE PRECISION,        -- watched level (reclaim/swing); null for compression
    range_high      DOUBLE PRECISION,        -- compression envelope
    range_low       DOUBLE PRECISION,
    atr             DOUBLE PRECISION,
    relaxed         BOOLEAN      DEFAULT FALSE,  -- momentum-relaxed eligible at arm
    ext_atr         DOUBLE PRECISION,            -- extension past EMA21 in ATRs (relaxed metric)
    armed_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    session_date    DATE         NOT NULL,       -- ET trading date the zone armed on
    outcome         TEXT         NOT NULL DEFAULT 'armed',  -- armed / fired / expired
    outcome_at      TIMESTAMPTZ,
    fired_signal_id UUID         REFERENCES signals(id) ON DELETE SET NULL
);

-- Lookup the open ('armed') row for a ticker+detector quickly on fire.
CREATE INDEX IF NOT EXISTS idx_azh_open
    ON armed_zone_history (ticker, detector, outcome, armed_at DESC);

-- Sweep stale rows + per-session analysis.
CREATE INDEX IF NOT EXISTS idx_azh_session
    ON armed_zone_history (session_date, outcome);

CREATE INDEX IF NOT EXISTS idx_azh_detector_date
    ON armed_zone_history (detector, session_date);

-- Service-role only (engine writes, admin reads via JWT-gated endpoint).
ALTER TABLE armed_zone_history ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'armed_zone_history' AND policyname = 'service_role_all'
    ) THEN
        CREATE POLICY service_role_all ON armed_zone_history
            FOR ALL TO service_role USING (true) WITH CHECK (true);
    END IF;
END $$;
