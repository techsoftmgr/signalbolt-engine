-- ── Signal Feedback Table ────────────────────────────────────────────────────
-- Run this in Supabase SQL Editor
-- Stores per-user thumbs up/down reactions on signal cards

CREATE TABLE IF NOT EXISTS signal_feedback (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  signal_id  UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
  user_id    UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  feedback   TEXT NOT NULL CHECK (feedback IN ('up', 'down')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (signal_id, user_id)   -- one vote per user per signal
);

-- Index for fast per-signal count queries
CREATE INDEX IF NOT EXISTS idx_signal_feedback_signal_id ON signal_feedback (signal_id);
CREATE INDEX IF NOT EXISTS idx_signal_feedback_user_id   ON signal_feedback (user_id);

-- Row Level Security
ALTER TABLE signal_feedback ENABLE ROW LEVEL SECURITY;

-- Users can read all feedback counts (for display)
CREATE POLICY "anyone_can_read_feedback"
  ON signal_feedback FOR SELECT
  USING (true);

-- Users can only insert/update/delete their own feedback
CREATE POLICY "users_manage_own_feedback"
  ON signal_feedback FOR ALL
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());
