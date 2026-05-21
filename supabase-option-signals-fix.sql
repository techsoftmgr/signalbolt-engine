-- ============================================================
-- Fix: add missing lifecycle columns to option_signals table
-- Run in: Supabase Dashboard → SQL Editor
-- Safe to re-run (uses IF NOT EXISTS / DO block)
-- ============================================================

ALTER TABLE option_signals
    ADD COLUMN IF NOT EXISTS result        TEXT    DEFAULT NULL,   -- 'win' | 'loss' | 'expired'
    ADD COLUMN IF NOT EXISTS closed_reason TEXT    DEFAULT NULL,   -- 'target_hit' | 'stop_hit' | 'expired'
    ADD COLUMN IF NOT EXISTS closed_at     TIMESTAMPTZ DEFAULT NULL;

-- Verify
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'option_signals'
  AND column_name IN ('result', 'closed_reason', 'closed_at')
ORDER BY column_name;
