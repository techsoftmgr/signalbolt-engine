-- Market Pulse Phase 2A — stalling-day detection. Run in the Supabase SQL editor.
-- Adds stalling counts + the combined "effective distribution" pressure metric.
-- Defaults keep existing rows valid; with zero stalling days the regime is unchanged.

alter table market_pulse_daily add column if not exists stall_count_spy  int          not null default 0;
alter table market_pulse_daily add column if not exists stall_count_qqq  int          not null default 0;
alter table market_pulse_daily add column if not exists effective_dd_spy numeric(4,1) not null default 0;
alter table market_pulse_daily add column if not exists effective_dd_qqq numeric(4,1) not null default 0;
