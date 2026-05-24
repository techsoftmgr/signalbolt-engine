-- Engine heartbeat table — worker writes here every 60s; /ready reads here
-- to surface "worker silently died" as a degraded readiness state.
--
-- Run in Supabase SQL Editor before the next deploy. The worker treats a
-- missing table as non-fatal (logs a warning), so this is safe to defer.

CREATE TABLE IF NOT EXISTS engine_heartbeats (
    service     TEXT PRIMARY KEY,
    last_beat   TIMESTAMPTZ NOT NULL DEFAULT now(),
    pid         INTEGER,
    machine_id  TEXT,
    notes       TEXT
);

-- Read-only for anon / authenticated; the engine writes via service key.
ALTER TABLE engine_heartbeats ENABLE ROW LEVEL SECURITY;

CREATE POLICY "engine_heartbeats_read_all"
    ON engine_heartbeats FOR SELECT
    USING (true);

-- No insert/update policy — only the service role (used by the engine) can
-- write, and service role bypasses RLS.
