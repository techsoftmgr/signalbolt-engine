-- ============================================================
-- Entry-gate rejection — detector tag
-- Records which detector produced the blocked candidate (SMC /
-- PULLBACK / COMPRESSION / SWING_BREAKOUT / EMA_RECLAIM / ...) so the
-- Gate Performance card + chart page can show the signal type.
--
-- Run in: Supabase Dashboard → SQL Editor (safe to re-run).
-- ============================================================

ALTER TABLE entry_gate_rejections
    ADD COLUMN IF NOT EXISTS detector TEXT;   -- SMC / PULLBACK / COMPRESSION / SWING_BREAKOUT / EMA_RECLAIM

CREATE INDEX IF NOT EXISTS idx_egr_detector
    ON entry_gate_rejections (detector, created_at);
