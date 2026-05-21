-- =============================================================================
-- SignalBolt — Lifecycle & Quality Migration
-- Run this in the Supabase SQL editor ONCE before deploying the upgraded engine.
--
-- What this adds:
--   1. New columns on `signals` table (confidence_grade, risk_grade, chop_score,
--      setup_type, missing_confirmations)
--   2. `setup_watchlist` table — staging area for WATCHLIST/DEVELOPING setups
--      before they promote to CONFIRMED_SIGNAL (written to `signals`)
--   3. `analytics_reports` table — daily report snapshots for the Analytics tab
-- =============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- 1. New columns on signals table
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE signals
  ADD COLUMN IF NOT EXISTS confidence_grade        VARCHAR(3)   DEFAULT 'B',
  ADD COLUMN IF NOT EXISTS risk_grade              VARCHAR(10)  DEFAULT 'MEDIUM',
  ADD COLUMN IF NOT EXISTS chop_score              FLOAT        DEFAULT 0,
  ADD COLUMN IF NOT EXISTS setup_type              VARCHAR(50)  DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS missing_confirmations   JSONB        DEFAULT '[]'::jsonb;

COMMENT ON COLUMN signals.confidence_grade      IS 'A+/A/B+/B/C based on final score bands';
COMMENT ON COLUMN signals.risk_grade            IS 'LOW/MEDIUM/HIGH composite risk assessment';
COMMENT ON COLUMN signals.chop_score            IS '0–100; chop detector score at time of signal (0 = clean)';
COMMENT ON COLUMN signals.setup_type            IS 'e.g. CHOCH_OB_RETEST, FVG_RETEST, VWAP_MEAN_REVERSION';
COMMENT ON COLUMN signals.missing_confirmations IS 'Array of strings describing what the setup still needs';


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. setup_watchlist — staging area for pre-signal setups
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS setup_watchlist (
  id                   UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
  ticker               VARCHAR(20)  NOT NULL,
  direction            VARCHAR(5)   NOT NULL,     -- LONG / SHORT
  strategy_type        VARCHAR(20)  NOT NULL,
  state                VARCHAR(20)  NOT NULL DEFAULT 'WATCHLIST',  -- WATCHLIST / DEVELOPING / CONFIRMED_SIGNAL / EXPIRED / INVALIDATED
  score                INTEGER      NOT NULL DEFAULT 0,
  confidence_grade     VARCHAR(3)   NOT NULL DEFAULT 'C',
  risk_grade           VARCHAR(10)  NOT NULL DEFAULT 'HIGH',
  setup_type           VARCHAR(50),
  regime_type          VARCHAR(20),
  session_mode         VARCHAR(20),
  chop_score           FLOAT        DEFAULT 0,
  entry_price          FLOAT,
  stop_loss            FLOAT,
  target_one           FLOAT,
  target_two           FLOAT,
  risk_reward          FLOAT,
  missing_confirmations JSONB       DEFAULT '[]'::jsonb,
  score_breakdown      JSONB        DEFAULT '{}'::jsonb,
  scan_count           INTEGER      DEFAULT 1,       -- how many scans this setup has been seen in
  first_seen_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  last_seen_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  promoted_at          TIMESTAMPTZ,                  -- when it became CONFIRMED_SIGNAL
  signal_id            UUID         REFERENCES signals(id) ON DELETE SET NULL,
  invalidation_reason  TEXT,
  created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  updated_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Index for fast per-ticker queries during scan cycles
CREATE INDEX IF NOT EXISTS idx_setup_watchlist_ticker_strategy
  ON setup_watchlist (ticker, strategy_type, state);

CREATE INDEX IF NOT EXISTS idx_setup_watchlist_state_updated
  ON setup_watchlist (state, updated_at);

-- Auto-update updated_at on any row change
CREATE OR REPLACE FUNCTION _update_setup_watchlist_ts()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_setup_watchlist_ts ON setup_watchlist;
CREATE TRIGGER trg_setup_watchlist_ts
  BEFORE UPDATE ON setup_watchlist
  FOR EACH ROW EXECUTE FUNCTION _update_setup_watchlist_ts();

COMMENT ON TABLE setup_watchlist IS
  'Staging area for setups progressing from WATCHLIST → DEVELOPING → CONFIRMED_SIGNAL. '
  'Only CONFIRMED_SIGNAL setups are promoted to the signals table.';


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. analytics_reports — daily report snapshots
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS analytics_reports (
  id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  report_date  DATE        NOT NULL UNIQUE,
  overall      JSONB       NOT NULL DEFAULT '{}'::jsonb,
  by_strategy  JSONB       NOT NULL DEFAULT '{}'::jsonb,
  quality_flags JSONB      NOT NULL DEFAULT '[]'::jsonb,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analytics_reports_date
  ON analytics_reports (report_date DESC);

COMMENT ON TABLE analytics_reports IS
  'Daily analytics snapshots: win rate, expectancy, R-multiples, false-positive flags. '
  'Written by the 5:30 PM ET analytics job, read by the Analytics tab.';


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Row-Level Security — ensure new tables are protected
-- ─────────────────────────────────────────────────────────────────────────────

-- setup_watchlist: engine writes (service role), app reads (anon for their own data)
ALTER TABLE setup_watchlist ENABLE ROW LEVEL SECURITY;

CREATE POLICY IF NOT EXISTS "Service role full access"
  ON setup_watchlist FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- analytics_reports: engine writes, authenticated users read
ALTER TABLE analytics_reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY IF NOT EXISTS "Service role full access"
  ON analytics_reports FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

CREATE POLICY IF NOT EXISTS "Authenticated users can read reports"
  ON analytics_reports FOR SELECT
  TO authenticated
  USING (true);

-- ─────────────────────────────────────────────────────────────────────────────
-- Done. Run `SELECT * FROM setup_watchlist LIMIT 0;` to verify.
-- ─────────────────────────────────────────────────────────────────────────────
