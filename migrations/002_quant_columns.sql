-- ═══════════════════════════════════════════════════════════════
--  SIGNALBOLT — QUANT SYSTEM MIGRATION
--  Adds quant metadata columns to the signals table.
--  Run this in your Supabase SQL editor.
-- ═══════════════════════════════════════════════════════════════

-- ── Quant columns on signals table ────────────────────────────
ALTER TABLE signals
  ADD COLUMN IF NOT EXISTS regime_type        TEXT    DEFAULT '',
  ADD COLUMN IF NOT EXISTS session_mode       TEXT    DEFAULT '',
  ADD COLUMN IF NOT EXISTS confidence_tier    TEXT    DEFAULT 'B',
  ADD COLUMN IF NOT EXISTS position_multiplier NUMERIC DEFAULT 1.0,
  ADD COLUMN IF NOT EXISTS gamma_net_gex      NUMERIC DEFAULT 0,
  ADD COLUMN IF NOT EXISTS gamma_is_negative  BOOLEAN DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS manipulation_clean BOOLEAN DEFAULT TRUE,
  ADD COLUMN IF NOT EXISTS manipulation_flags TEXT[]  DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS sl_adjustments     TEXT[]  DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS risk_reward        NUMERIC DEFAULT 0,
  ADD COLUMN IF NOT EXISTS score_breakdown    JSONB   DEFAULT '{}';

-- ── Indexes for common filter queries ─────────────────────────
CREATE INDEX IF NOT EXISTS idx_signals_regime        ON signals(regime_type);
CREATE INDEX IF NOT EXISTS idx_signals_tier          ON signals(confidence_tier);
CREATE INDEX IF NOT EXISTS idx_signals_session       ON signals(session_mode);
CREATE INDEX IF NOT EXISTS idx_signals_rr            ON signals(risk_reward DESC);
CREATE INDEX IF NOT EXISTS idx_signals_score_breakdown ON signals USING gin(score_breakdown);

-- ── Backfill existing signals with sensible defaults ──────────
UPDATE signals
SET
  confidence_tier     = CASE
    WHEN confidence_score >= 90 THEN 'A+'
    WHEN confidence_score >= 80 THEN 'A'
    WHEN confidence_score >= 70 THEN 'B+'
    ELSE 'B'
  END,
  position_multiplier = CASE
    WHEN confidence_score >= 90 THEN 1.0
    WHEN confidence_score >= 80 THEN 0.75
    WHEN confidence_score >= 70 THEN 0.5
    ELSE 0.25
  END,
  regime_type         = 'TRENDING_BULL',
  session_mode        = 'STANDARD',
  manipulation_clean  = TRUE,
  risk_reward         = CASE
    WHEN entry_price > 0 AND stop_loss > 0 AND target_one > 0
    THEN ROUND(ABS(target_one - entry_price) / NULLIF(ABS(entry_price - stop_loss), 0), 2)
    ELSE 2.0
  END
WHERE confidence_tier IS NULL OR confidence_tier = '';

-- ═══════════════════════════════════════════════════════════════
--  MARKET REGIME SNAPSHOTS TABLE
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS market_regime_snapshots (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  regime_type     TEXT    NOT NULL,
  vix             NUMERIC NOT NULL,
  vix_change_pct  NUMERIC NOT NULL DEFAULT 0,
  adx             NUMERIC NOT NULL DEFAULT 0,
  above_200ma     BOOLEAN NOT NULL DEFAULT TRUE,
  spy_price       NUMERIC NOT NULL DEFAULT 0,
  fear_greed      INTEGER NOT NULL DEFAULT 50,
  captured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_regime_snapshots_ts ON market_regime_snapshots(captured_at DESC);

-- ── RLS for market_regime_snapshots ──────────────────────────
-- Engine writes via service role key (bypasses RLS).
-- App reads via anon/auth key — allow SELECT only.
ALTER TABLE market_regime_snapshots ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read regime snapshots"
  ON market_regime_snapshots
  FOR SELECT
  USING (true);

-- No INSERT/UPDATE/DELETE policy for anon/auth — only service role can write.

-- Enable realtime (app can subscribe to live regime changes)
-- Wrapped in DO block so re-running is safe if already a member
DO $$
BEGIN
  ALTER PUBLICATION supabase_realtime ADD TABLE market_regime_snapshots;
EXCEPTION WHEN OTHERS THEN
  NULL; -- already a member of the publication, ignore
END;
$$;

-- ═══════════════════════════════════════════════════════════════
--  SELF-LEARNING WEIGHTS TABLE
-- ═══════════════════════════════════════════════════════════════

-- Populated weekly by weight_optimizer.py (runs via service role key).
-- regime_type 'ANY' = weight set applies to all regimes.
CREATE TABLE IF NOT EXISTS signal_weights (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  strategy_type  TEXT        NOT NULL,
  regime_type    TEXT        NOT NULL DEFAULT 'ANY',
  weights        JSONB       NOT NULL,
  metrics        JSONB       NOT NULL DEFAULT '{}',
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(strategy_type, regime_type)
);

CREATE INDEX IF NOT EXISTS idx_signal_weights_strategy ON signal_weights(strategy_type);

-- ── RLS for signal_weights ────────────────────────────────────
-- App/clients: read-only (so the app can display "last optimized" info).
-- Writes: service role key only (optimizer runs server-side).
ALTER TABLE signal_weights ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read signal weights"
  ON signal_weights
  FOR SELECT
  USING (true);

-- No INSERT/UPDATE/DELETE policy for anon/auth — optimizer uses service key.

-- ── Seed default weights (optimizer overwrites after first run) ───────────
INSERT INTO signal_weights (strategy_type, regime_type, weights, metrics)
VALUES
  ('scalping',     'ANY', '{"smc":15,"technical":40,"sentiment":15,"risk":30,"l5_bonus":0,"l6_bonus":8,"l7_bonus":6,"l8_bonus":8,"l9_bonus":8}', '{"note":"default"}'),
  ('day_trade',    'ANY', '{"smc":25,"technical":35,"sentiment":25,"risk":15,"l5_bonus":5,"l6_bonus":8,"l7_bonus":6,"l8_bonus":8,"l9_bonus":8}', '{"note":"default"}'),
  ('swing_trade',  'ANY', '{"smc":40,"technical":30,"sentiment":20,"risk":10,"l5_bonus":8,"l6_bonus":8,"l7_bonus":6,"l8_bonus":8,"l9_bonus":8}', '{"note":"default"}'),
  ('options_flow', 'ANY', '{"smc":10,"technical":20,"sentiment":50,"risk":20,"l5_bonus":0,"l6_bonus":8,"l7_bonus":6,"l8_bonus":8,"l9_bonus":8}', '{"note":"default"}'),
  ('dark_pool',    'ANY', '{"smc":10,"technical":20,"sentiment":60,"risk":10,"l5_bonus":0,"l6_bonus":8,"l7_bonus":6,"l8_bonus":8,"l9_bonus":8}', '{"note":"default"}')
ON CONFLICT (strategy_type, regime_type) DO NOTHING;
