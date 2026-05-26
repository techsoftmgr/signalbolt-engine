-- Purge cancelled signal rows from the DB.
--
-- Background: the earlier dedup cleanup SQL set result='cancelled' on every
-- duplicate active signal it consolidated. Those rows are noise — they were
-- never real trades — and they leave grey "Closed" cards on the Signals tab.
-- This deletes them outright so they don't show in either Signals OR History.
--
-- Preview (run as SELECT first if you want a count before deleting):
--   SELECT COUNT(*) AS will_delete FROM signals WHERE result = 'cancelled';

DELETE FROM signals
WHERE result = 'cancelled';

-- option_signals has no 'cancelled' result column convention right now —
-- the cleanup SQL just set status='closed' without a result marker, so
-- there's nothing reliable to filter on. Skipping intentionally.
