-- Agreement track record — logs each chart-read snapshot (Technical vs Quant
-- verdict) so we can later measure WHICH method is right when they disagree.
--
-- The engine writes one row per ticker per day (best-effort). A scheduled scorer
-- fills the forward-outcome columns after the horizon elapses; an admin view then
-- reports "when TA & Quant disagree, TA was right X% / Quant Y%".
--
-- Service-role only (RLS on, no public policy) — telemetry, not user data.
-- Idempotent / safe to re-run.

CREATE TABLE IF NOT EXISTS chart_read_log (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  ticker             text NOT NULL,
  ta_bias            text NOT NULL,            -- bullish | bearish | neutral
  quant_bias         text,                     -- bullish | bearish | neutral | null
  agreement          text NOT NULL,            -- agree | disagree | partial | n/a
  short_term         text,                     -- confirming | diverging | neutral
  price              numeric NOT NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  -- filled later by the forward-outcome scorer:
  horizon_days       int,
  forward_price      numeric,
  forward_return_pct numeric,                  -- signed % move over the horizon
  winner             text,                     -- ta | quant | both | neither | null
  scored_at          timestamptz
);

-- One snapshot per ticker per day (the logger upserts on this).
CREATE UNIQUE INDEX IF NOT EXISTS uq_chart_read_log_ticker_day
  ON chart_read_log (ticker, (created_at::date));

-- Fast lookups for the scorer (unscored rows past their horizon) + reads.
CREATE INDEX IF NOT EXISTS idx_chart_read_log_unscored
  ON chart_read_log (created_at) WHERE winner IS NULL;
CREATE INDEX IF NOT EXISTS idx_chart_read_log_ticker
  ON chart_read_log (ticker, created_at DESC);

ALTER TABLE chart_read_log ENABLE ROW LEVEL SECURITY;
-- (No public policy → only the service role can read/write. Admin endpoints use it.)
