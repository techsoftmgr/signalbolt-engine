-- Clean up existing duplicate active signals before applying the unique index.
--
-- Strategy: for each (ticker, strategy_type) with multiple active rows, keep
-- the OLDEST one (the "winner" that fired first) and DELETE the rest. They
-- were never real trades — just engine-race artifacts — so there's no audit
-- value in keeping them, and a 'cancelled' marker would just clutter the
-- Signals tab with grey "Closed" cards.

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
DELETE FROM signals
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);

-- Same for option_signals — delete duplicate active rows, keep oldest.
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
DELETE FROM option_signals
WHERE id IN (SELECT id FROM ranked_opts WHERE rn > 1);

-- Verify (run these as SELECTs first if you want to preview):
-- SELECT ticker, strategy_type, COUNT(*) FROM signals WHERE status='active'
--   GROUP BY ticker, strategy_type HAVING COUNT(*) > 1;
-- Should return 0 rows.
