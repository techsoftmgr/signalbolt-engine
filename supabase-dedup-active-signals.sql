-- Race-condition fix: enforce one active signal per (ticker, strategy_type).
--
-- Symptom we're fixing:
--   Two parallel scans (APScheduler day_trade_10min job + stream.py bar-close
--   event) both call _has_active_signal() within a few ms. Both see no signal,
--   both compute the same setup, both INSERT. Result: identical day_trade
--   signals for the same ticker, IDs differ by ~0-4 seconds.
--
-- This partial unique index makes the second INSERT fail atomically with
-- 23505 (unique_violation). The engine's _write_signal catches that and
-- logs it instead of crashing.
--
-- Why partial (WHERE status = 'active'):
--   We want any number of historical 'closed' signals for the same ticker,
--   but only ONE 'active' at a time. A full unique constraint would block
--   ever re-signaling the ticker.

CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_active_signal
ON signals (ticker, strategy_type)
WHERE status = 'active';

-- Same protection for option_signals — same race exists on that table.
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_active_option_signal
ON option_signals (ticker, status)
WHERE status = 'active';
