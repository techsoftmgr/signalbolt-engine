# Database Migrations

All schema changes live here as numbered SQL files.
**Never edit an existing file — always add a new one.**

## Files

| File | Description | Status |
|---|---|---|
| `001_initial_schema.sql` | Core tables: profiles, signals, subscriptions | ✅ Applied to prod |
| `002_quant_columns.sql` | Quant columns on signals + market_regime_snapshots + signal_weights | ⏳ Apply to prod |
| `003_signal_events.sql` | signal_events table + trigger fn_signal_fired | ⏳ Apply to prod |

## How to apply a migration

1. Open [Supabase SQL Editor](https://supabase.com/dashboard)
2. Select the correct project (dev or prod)
3. Paste the SQL file contents and run it
4. Mark it as applied in this table

## How to create a new migration

1. Create `migrations/004_description.sql`
2. Apply it to **dev Supabase first**
3. Test your change locally or on Railway dev
4. Open a PR → merge to main
5. Apply the **same file** to **prod Supabase**

## Environments

| Environment | Supabase Project | URL |
|---|---|---|
| DEV | `signalbolt-dev` | Create at app.supabase.com |
| PROD | `signalbolt` | https://hjfgfeytefituywnopwt.supabase.co |
