-- Manual override / management-mode for stock + option signals.
--
-- Adds two columns to BOTH signals and option_signals:
--   management_mode : 'engine' (default) | 'manual'
--       'engine' → the engine manages the signal normally (trail, EOD close,
--                  RT stop/target, reversal exit, etc.)
--       'manual' → the engine does NOT touch it. Stop/targets/status only change
--                  when the admin updates them. The signal is "frozen" to the
--                  engine until flipped back to 'engine'.
--   origin          : 'engine' (default) | 'manual'  — who created the signal.
--
-- Existing rows get 'engine' for both, so behaviour is unchanged for everything
-- the engine has already created. Idempotent — safe to re-run.

ALTER TABLE signals
  ADD COLUMN IF NOT EXISTS management_mode text NOT NULL DEFAULT 'engine',
  ADD COLUMN IF NOT EXISTS origin          text NOT NULL DEFAULT 'engine';

ALTER TABLE option_signals
  ADD COLUMN IF NOT EXISTS management_mode text NOT NULL DEFAULT 'engine',
  ADD COLUMN IF NOT EXISTS origin          text NOT NULL DEFAULT 'engine';

-- Helps the 5-min / RT managers cheaply filter "active + engine-managed".
CREATE INDEX IF NOT EXISTS idx_signals_status_mgmt
  ON signals (status, management_mode);
CREATE INDEX IF NOT EXISTS idx_option_signals_status_mgmt
  ON option_signals (status, management_mode);

-- Optional sanity constraint (kept permissive; comment out if undesired).
-- ALTER TABLE signals        ADD CONSTRAINT chk_signals_mgmt        CHECK (management_mode IN ('engine','manual'));
-- ALTER TABLE option_signals ADD CONSTRAINT chk_option_signals_mgmt CHECK (management_mode IN ('engine','manual'));
