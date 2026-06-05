-- Per-job run ledger — ONE upserted row per scheduled job_id, written by the
-- APScheduler execution listener on the worker. Powers the Market-tab "Daily
-- Jobs" report (served from the web process; this table is the cross-process
-- bridge). Run once in the Supabase SQL editor.

create table if not exists job_runs (
  job_id            text primary key,
  last_started      timestamptz,
  last_finished     timestamptz,
  last_status       text,          -- success | error | missed
  last_duration_ms  integer,
  last_summary      text,          -- short human line if the job returned one
  last_error        text,
  run_count         integer default 0,
  error_count       integer default 0,
  updated_at        timestamptz not null default now()
);

create index if not exists idx_job_runs_updated on job_runs (updated_at desc);

alter table job_runs enable row level security;

-- Read-only to clients for now (the report is "open for all users"); the worker
-- writes via the service-role key which bypasses RLS.
do $$ begin
  if not exists (select 1 from pg_policies where tablename='job_runs' and policyname='job_runs_read') then
    create policy job_runs_read on job_runs for select using (true);
  end if;
end $$;
