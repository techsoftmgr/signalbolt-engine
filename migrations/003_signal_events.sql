-- signal_events: tracks lifecycle events for each signal
CREATE TABLE IF NOT EXISTS signal_events (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  signal_id   uuid        NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
  event_type  text        NOT NULL,
  -- event_type values: 'fired', 't1_hit', 'be_move', 'closed_win', 'closed_loss',
  --                    'market_close', 'reversal', 'time_limit', 'option_expired'
  price       numeric,        -- stock price at time of event (optional)
  note        text,           -- human-readable description shown in the app
  created_at  timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS signal_events_signal_id_idx ON signal_events(signal_id);
CREATE INDEX IF NOT EXISTS signal_events_created_at_idx ON signal_events(created_at DESC);

ALTER TABLE signal_events ENABLE ROW LEVEL SECURITY;
-- Read-only for all authenticated users (same as signals table)
DO $$
BEGIN
  CREATE POLICY "signal_events_read" ON signal_events
    FOR SELECT USING (true);
EXCEPTION WHEN OTHERS THEN
  NULL; -- policy already exists, ignore
END;
$$;

CREATE OR REPLACE FUNCTION fn_signal_fired()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  INSERT INTO signal_events(signal_id, event_type, price, note)
  VALUES (
    NEW.id,
    'fired',
    NEW.entry_price,
    'Signal fired — ' || NEW.direction || ' ' || NEW.ticker || ' @ $' || NEW.entry_price
  );
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_signal_fired ON signals;
CREATE TRIGGER trg_signal_fired
  AFTER INSERT ON signals
  FOR EACH ROW EXECUTE FUNCTION fn_signal_fired();
