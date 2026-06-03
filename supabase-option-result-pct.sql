-- Add realized-premium P&L columns to option_signals (parity with signals).
-- The option monitor + admin close now record these on every option close so
-- PUT/CALL expectancy is measurable (was win/loss-only). Run once in the
-- Supabase SQL editor. Safe/idempotent.

alter table public.option_signals
    add column if not exists result_pct numeric,
    add column if not exists result_pnl numeric;
