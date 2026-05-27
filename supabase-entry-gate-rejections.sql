-- Entry Gate v2 rejection telemetry
--
-- Captures every signal that scored well enough to fire (passed scorer +
-- chop + manipulation gates) but was BLOCKED by entry_gate. Lets us
-- analyse later whether the gate was correctly rejecting losers, or
-- accidentally rejecting winners.
--
-- Apply once in Supabase SQL editor.

CREATE TABLE IF NOT EXISTS entry_gate_rejections (
    id                BIGSERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ticker            VARCHAR(20) NOT NULL,
    direction         VARCHAR(10) NOT NULL,
    strategy_type     VARCHAR(20) NOT NULL,
    price             NUMERIC(12, 4),
    confidence_score  NUMERIC(6, 2),
    -- Per-gate result: {"15m_trend":"pass","5m_macd":"fail: ...","1m_reversal":"pass","patterns":"pass"}
    gate_log          JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Flat array of human-readable failure reasons (1 per failed gate)
    reasons           TEXT[] NOT NULL DEFAULT '{}',
    -- For optional outcome-checking: did the rejection turn out to be right?
    -- (filled in by a future backtest job — leave NULL on insert)
    would_have_won    BOOLEAN,
    realized_pnl_pct  NUMERIC(8, 4)
);

CREATE INDEX IF NOT EXISTS idx_egr_created_at ON entry_gate_rejections (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_egr_ticker     ON entry_gate_rejections (ticker);
CREATE INDEX IF NOT EXISTS idx_egr_strategy   ON entry_gate_rejections (strategy_type);

COMMENT ON TABLE entry_gate_rejections IS
  'Signals blocked by engine/entry_gate.py. Used to validate whether the gate is rejecting losers (good) or winners (bad).';

-- ── Row-Level Security ──────────────────────────────────────────────
-- This table is for internal engine telemetry only. End users should
-- never read or write it. Service role (engine + worker) bypasses RLS
-- automatically, so writes from runner.py continue to work.
ALTER TABLE entry_gate_rejections ENABLE ROW LEVEL SECURITY;

-- No policies = no access for anon / authenticated roles.
-- (Service role bypasses RLS, so the engine still writes rows fine.)
--
-- If you ever want to expose this to an admin dashboard in the app, add:
--   CREATE POLICY "admin_read" ON entry_gate_rejections
--     FOR SELECT TO authenticated
--     USING (auth.jwt() ->> 'email' = 'techsoftmgr@gmail.com');
