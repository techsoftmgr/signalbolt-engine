-- ============================================================
-- Entry-gate rejection — full lifecycle columns
-- The validator already backfills would_have_won + realized_pnl_pct.
-- These columns persist the computed trade geometry (stop/target) and
-- the simulated exit so the admin Gate Performance view can show
-- entry / stop / exit per rejection, like Detector Performance.
--
-- Run in: Supabase Dashboard → SQL Editor (safe to re-run).
-- ============================================================

ALTER TABLE entry_gate_rejections
    ADD COLUMN IF NOT EXISTS stop_loss   DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS target_one  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS exit_price  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS exit_reason TEXT;   -- target_hit / stop_hit / window_end
