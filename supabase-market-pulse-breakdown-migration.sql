-- Market Pulse — breadth BREAKDOWN (bearish mirror of the thrust). Run in Supabase.
-- The 10-day breadth EMA collapsing from >0.615 to <0.40 within 10 sessions.

alter table market_pulse_daily add column if not exists breadth_breakdown boolean not null default false;
