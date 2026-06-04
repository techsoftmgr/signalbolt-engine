-- Fundamentals quality-screen cache (SEC EDGAR). One row per ticker; the engine
-- rolling-refreshes the stalest names (fundamentals change quarterly, so a few
-- refreshes/day is plenty). Service-role only. Run once in the SQL editor.

create table if not exists public.fundamentals_cache (
    ticker          text primary key,
    net_margin      numeric,
    roe             numeric,
    debt_to_equity  numeric,
    revenue_growth  numeric,
    fcf_positive    boolean,
    quality_score   integer,
    metrics         jsonb,
    fetched_at      timestamptz not null default now()
);

alter table public.fundamentals_cache enable row level security;
-- (No public policy → engine/service-role only. Admin endpoint reads it.)
