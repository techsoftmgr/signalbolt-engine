-- Market-regime timeline (intraday transitions across pre-market / RTH / after-hours).
-- The engine COMPUTES the regime continuously (regime_detector) but never logged it,
-- so intraday regime-at-fire / regime-during-hold couldn't be reconstructed. This
-- table records the full daily transition (write-on-change), e.g.:
--   05:00 pre  TRENDING_BULL → 08:30 pre PANIC → 12:00 rth TRENDING_BULL → 14:00 rth RANGING
-- Powers: exact intraday regime-at-fire, regime-during-hold studies, regime-conditional
-- detector enablement/sizing, and a "market regime today" timeline UI.
--
-- Run once in the Supabase SQL editor.

create table if not exists regime_history (
  id              uuid primary key default gen_random_uuid(),
  captured_at     timestamptz not null default now(),
  regime_type     text not null,              -- PANIC | HIGH_VOL | RISK_OFF | TRENDING_BEAR | RANGING | TRENDING_BULL | LOW_VOL
  session         text,                       -- pre | rth | post | closed
  vix             numeric,
  vix_change_pct  numeric,                    -- intraday VIX change (fraction, e.g. 0.18 = +18%)
  adx             numeric,
  above_200ma     boolean,
  spy_price       numeric,
  fear_greed      integer,
  blocked         boolean default false,      -- new entries paused (PANIC / VIX spike)
  note            text                        -- e.g. "extended-hours VIX is approximate"
);

create index if not exists idx_regime_history_captured on regime_history (captured_at desc);

-- Service-role only: the engine (service key) writes + reads; the app reads via the
-- engine /market/regime-history endpoint (regime isn't user-sensitive, but keeping
-- it server-side mirrors /indices).
alter table regime_history enable row level security;
