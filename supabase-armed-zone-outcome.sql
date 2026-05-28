-- ============================================================
-- Armed-zone counterfactual outcome
-- For zones that armed but never fired, record whether a breakout entry
-- WOULD have won — so we can tell if the firing filters (retest / volume /
-- regime) are correctly skipping losers or killing winners.
--
--   would_have_won : true/false (null = undetermined / never triggered)
--   realized_pnl_pct: simulated P/L vs the entry level
--   outcome can now also be 'expired_no_trigger' (price never crossed the level)
--
-- Run in: Supabase Dashboard → SQL Editor (safe to re-run).
-- ============================================================

ALTER TABLE armed_zone_history
    ADD COLUMN IF NOT EXISTS would_have_won    BOOLEAN,
    ADD COLUMN IF NOT EXISTS realized_pnl_pct  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sim_entry         DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sim_stop          DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sim_target        DOUBLE PRECISION;
