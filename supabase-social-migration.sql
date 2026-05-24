-- ─────────────────────────────────────────────────────────────────────────────
-- SignalBolt — Social / Community Feature Migration
-- Run once in Supabase SQL Editor (Dashboard → SQL Editor → New query)
-- Safe to re-run: all statements use IF NOT EXISTS / ON CONFLICT DO NOTHING
-- ─────────────────────────────────────────────────────────────────────────────

-- ── signal_votes ─────────────────────────────────────────────────────────────
-- One vote per user per signal (upsert with on_conflict="signal_id,user_id")
CREATE TABLE IF NOT EXISTS signal_votes (
    signal_id  UUID        NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    user_id    UUID        NOT NULL,
    vote_type  TEXT        NOT NULL CHECK (vote_type IN ('bullish', 'bearish', 'watching')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (signal_id, user_id)
);

-- ── signal_comments ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signal_comments (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id  UUID        NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    user_id    UUID        NOT NULL,
    content    TEXT        NOT NULL CHECK (char_length(content) BETWEEN 3 AND 500),
    is_flagged BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── signal_follows ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signal_follows (
    signal_id  UUID        NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    user_id    UUID        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (signal_id, user_id)
);

-- ── Indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_signal_votes_signal    ON signal_votes(signal_id);
CREATE INDEX IF NOT EXISTS idx_signal_votes_user      ON signal_votes(user_id);
CREATE INDEX IF NOT EXISTS idx_signal_comments_signal ON signal_comments(signal_id);
CREATE INDEX IF NOT EXISTS idx_signal_comments_user   ON signal_comments(user_id);
CREATE INDEX IF NOT EXISTS idx_signal_follows_signal  ON signal_follows(signal_id);
CREATE INDEX IF NOT EXISTS idx_signal_follows_user    ON signal_follows(user_id);

-- ── Row Level Security ────────────────────────────────────────────────────────
-- Votes: anyone can read; only the owner can insert/update their own row
ALTER TABLE signal_votes    ENABLE ROW LEVEL SECURITY;
ALTER TABLE signal_comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE signal_follows  ENABLE ROW LEVEL SECURITY;

-- Use service role key in the engine (bypasses RLS) — these policies guard direct DB access
DROP POLICY IF EXISTS "votes_read_all"   ON signal_votes;
DROP POLICY IF EXISTS "votes_own_write"  ON signal_votes;
DROP POLICY IF EXISTS "comments_read_public" ON signal_comments;
DROP POLICY IF EXISTS "comments_own_write"   ON signal_comments;
DROP POLICY IF EXISTS "follows_read_all"     ON signal_follows;
DROP POLICY IF EXISTS "follows_own_write"    ON signal_follows;

CREATE POLICY "votes_read_all"
    ON signal_votes FOR SELECT USING (true);
CREATE POLICY "votes_own_write"
    ON signal_votes FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "comments_read_public"
    ON signal_comments FOR SELECT USING (is_flagged = false);
CREATE POLICY "comments_own_write"
    ON signal_comments FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "follows_read_all"
    ON signal_follows FOR SELECT USING (true);
CREATE POLICY "follows_own_write"
    ON signal_follows FOR ALL USING (auth.uid() = user_id);
