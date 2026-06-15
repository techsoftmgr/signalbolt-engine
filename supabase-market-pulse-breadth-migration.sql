-- Market Pulse Phase 3 — breadth thrust. Run in the Supabase SQL editor.
-- Zweig-style breadth thrust derived from the advancers/decliners we already store.

alter table market_pulse_daily add column if not exists breadth_ratio  numeric(5,4);  -- 10-day EMA of adv/(adv+decl)
alter table market_pulse_daily add column if not exists breadth_thrust boolean not null default false;
