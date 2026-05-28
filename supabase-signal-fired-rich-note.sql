-- Dedup the 'fired' signal_event + make it rich.
--
-- Previously TWO 'fired' events were created per signal:
--   1. DB trigger trg_signal_fired → terse "Signal fired — LONG CCL @ $28.14"
--   2. runner._write_signal manual insert → rich "🟢 LONG signal fired @ $28.14
--      — 94% conviction (Day Trade) · Regime: ... · Target ... · Stop ..."
--
-- Fix: the manual insert is removed from runner.py, and this migration
-- upgrades the trigger to produce the rich note itself — so there is exactly
-- ONE 'fired' event per signal, covering ALL insert paths (engine, test
-- injection, etc.) without app-side duplication.
--
-- Apply once in the Supabase SQL editor.

CREATE OR REPLACE FUNCTION fn_signal_fired()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  INSERT INTO signal_events(signal_id, event_type, price, note)
  VALUES (
    NEW.id,
    'fired',
    NEW.entry_price,
    '🟢 ' || NEW.direction || ' signal fired @ $'
      || to_char(NEW.entry_price, 'FM999990.00')
      || ' — ' || COALESCE(NEW.confidence_score::text, '?') || '% conviction ('
      || initcap(replace(COALESCE(NEW.strategy_type, 'day_trade'), '_', ' ')) || ')'
      || CASE WHEN COALESCE(NEW.regime_type, '') <> ''
              THEN ' · Regime: ' || NEW.regime_type ELSE '' END
      || CASE WHEN NEW.target_one IS NOT NULL
              THEN ' · Target $' || to_char(NEW.target_one, 'FM999990.00') ELSE '' END
      || CASE WHEN NEW.stop_loss IS NOT NULL
              THEN ' · Stop $' || to_char(NEW.stop_loss, 'FM999990.00') ELSE '' END
  );
  RETURN NEW;
END;
$$;

-- Trigger definition unchanged (AFTER INSERT). Re-assert for idempotency.
DROP TRIGGER IF EXISTS trg_signal_fired ON signals;
CREATE TRIGGER trg_signal_fired
  AFTER INSERT ON signals
  FOR EACH ROW EXECUTE FUNCTION fn_signal_fired();
