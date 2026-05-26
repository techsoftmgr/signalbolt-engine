-- Clean up existing duplicate active signals before applying the unique index.
--
-- Strategy: for each (ticker, strategy_type) with multiple active rows,
-- keep the OLDEST one (the "winner" that fired first) and close the rest
-- as 'cancelled' with a note. This preserves audit trail while clearing
-- the dupes so the partial unique index can be created.

WITH ranked AS (
  SELECT
    id,
    ticker,
    strategy_type,
    created_at,
    ROW_NUMBER() OVER (
      PARTITION BY ticker, strategy_type
      ORDER BY created_at ASC, id ASC
    ) AS rn
  FROM signals
  WHERE status = 'active'
)
UPDATE signals
SET
  status     = 'closed',
  result     = 'cancelled',
  closed_at  = NOW(),
  result_pnl = 0
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);

-- Same for option_signals (used WHERE status='active' so any 'closed'
-- historical duplicates aren't touched).
WITH ranked_opts AS (
  SELECT
    id,
    ticker,
    created_at,
    ROW_NUMBER() OVER (
      PARTITION BY ticker
      ORDER BY created_at ASC, id ASC
    ) AS rn
  FROM option_signals
  WHERE status = 'active'
)
UPDATE option_signals
SET
  status    = 'closed',
  closed_at = NOW()
WHERE id IN (SELECT id FROM ranked_opts WHERE rn > 1);

-- Verify (run these as SELECTs first if you want to preview):
-- SELECT ticker, strategy_type, COUNT(*) FROM signals WHERE status='active'
--   GROUP BY ticker, strategy_type HAVING COUNT(*) > 1;
-- Should return 0 rows.
