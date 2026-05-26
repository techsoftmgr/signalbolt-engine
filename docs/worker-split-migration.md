# Migration: split worker into its own Fly app

Goal: stop Fly from running port-8000 health checks against worker machines,
and remove the routing ambiguity of a single app with two process types.

After this migration:
- `signalbolt-engine` — web only, scales horizontally
- `signalbolt-worker` — single machine, runs APScheduler + Alpaca WS

## One-time setup (you run these)

```powershell
$fly = "$env:USERPROFILE\.fly\bin\fly.exe"

# 1. Create the worker app
& $fly apps create signalbolt-worker --org personal

# 2. Mirror every secret from engine → worker.
#    First, list what's set on engine:
& $fly secrets list -a signalbolt-engine
#    Then set each one on worker. Examples (replace with your real values):
& $fly secrets set ALPACA_API_KEY="..." -a signalbolt-worker
& $fly secrets set ALPACA_SECRET_KEY="..." -a signalbolt-worker
& $fly secrets set SUPABASE_URL="..." -a signalbolt-worker
& $fly secrets set SUPABASE_SERVICE_KEY="..." -a signalbolt-worker
& $fly secrets set ANTHROPIC_API_KEY="..." -a signalbolt-worker
& $fly secrets set REDIS_URL="..." -a signalbolt-worker
# (and any others returned by `secrets list`)

# 3. First deploy of the worker app
& $fly deploy -c fly.worker.toml -a signalbolt-worker

# 4. Confirm worker came up cleanly
& $fly logs --no-tail -a signalbolt-worker | Select-String "engine|worker|started"

# 5. Redeploy the engine (now web-only)
& $fly deploy -a signalbolt-engine

# 6. Verify no second worker is left on the engine app
& $fly status -a signalbolt-engine
# You should see ONLY `app` process machines, no `worker` row.
```

## Rollback (if anything breaks)

```powershell
# Re-add worker process to the engine app temporarily:
#   - revert fly.toml to include `worker = "python -m engine.worker"` in [processes]
#   - redeploy engine
# Then destroy the worker app:
& $fly apps destroy signalbolt-worker -y
```

## Notes

- The worker app uses the same Dockerfile — no separate image build needed.
- Keep worker scale at exactly **1** machine. Two workers will both
  subscribe to Alpaca SIP and double-fire every push notification.
- Heartbeats still write to the `engine_heartbeats` Supabase table; the
  engine's `/ready` endpoint will continue to read them. No code change
  needed there because the heartbeat is keyed by service name, not host.
