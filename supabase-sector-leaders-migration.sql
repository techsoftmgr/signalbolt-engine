-- Sector Leaders (Phase 2C) — run in the Supabase SQL editor.
-- Daily RS ranking of the 11 S&P 500 sector SPDR ETFs vs SPY. Free/public read.

create table if not exists sector_leaders_daily (
  date            date    not null,
  sector_etf      text    not null,             -- 'XLK', 'XLF', ...
  rs_1m           numeric,
  rs_3m           numeric,
  rs_6m           numeric,
  rs_blended      numeric,
  rs_rank         int,                           -- 1..11 (1 = strongest)
  rs_rank_5d_ago  int,
  rank_momentum   text,                          -- 'IMPROVING' | 'DETERIORATING' | 'FLAT'
  above_50d       boolean,
  tilt            text,                          -- 'OFFENSE' | 'DEFENSE' | 'CYCLICAL'
  created_at      timestamptz not null default now(),
  primary key (date, sector_etf)
);

create table if not exists sector_leaders_summary (
  date            date primary key,
  tape_character  text    not null,              -- 'OFFENSE_LED' | 'DEFENSE_LED' | 'ROTATING'
  top3            text[]  not null,
  guidance_key    text    not null,
  created_at      timestamptz not null default now()
);

-- Public read (free feature); writes are service-role only (the daily job).
alter table sector_leaders_daily   enable row level security;
alter table sector_leaders_summary enable row level security;

drop policy if exists sector_leaders_daily_read   on sector_leaders_daily;
drop policy if exists sector_leaders_summary_read on sector_leaders_summary;
create policy sector_leaders_daily_read   on sector_leaders_daily   for select using (true);
create policy sector_leaders_summary_read on sector_leaders_summary for select using (true);

create index if not exists sector_leaders_daily_date_idx   on sector_leaders_daily (date desc);
create index if not exists sector_leaders_daily_sector_idx on sector_leaders_daily (sector_etf, date desc);
