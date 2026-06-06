"""
Job registry — the human-readable catalog of every scheduled job the engine
runs, grouped by cadence. Pairs with `job_runs` (the per-job last-run ledger
written by the APScheduler listener) to power the Market-tab "Daily Jobs" report.

Pure data + merge helpers (no I/O) → unit-testable. Keep `JOBS` in sync when a
new add_job() lands in runner.py (the listener records ANY job id automatically,
but unknown ids show with a generic label until added here).
"""
from __future__ import annotations

# cadence buckets for the report UI
INTRADAY = "intraday"   # polls through the session
DAILY    = "daily"      # once per trading day
SESSION  = "session"    # pre-market / open window
WEEKLY   = "weekly"     # weekly

# id → label / cadence / schedule (human) / what it does / category
JOBS: list[dict] = [
    # ── intraday pollers ──
    {"id": "maintenance",        "label": "Maintenance",            "cadence": INTRADAY, "schedule": "every 15 min, 24/7",       "category": "Health",   "what": "Housekeeping: expire stale signals, refresh trackers, prune state."},
    {"id": "eod_monitor",        "label": "Signal Monitor",         "cadence": INTRADAY, "schedule": "every 5 min, 9:30a–4:05p ET","category": "Signals",  "what": "Manages every active signal — stop/target/trail checks + MFE/MAE capture."},
    {"id": "day_trade_10min",    "label": "Day-Trade Scan",         "cadence": INTRADAY, "schedule": "every 10 min, RTH",         "category": "Signals",  "what": "Day-trade strategy scan."},
    {"id": "breakout_watch_sync","label": "Breakout-Watch Sync",    "cadence": INTRADAY, "schedule": "every 5 min, RTH",          "category": "Quant",    "what": "Advances the Breakout-Watch lifecycle (watching→triggered/faded)."},
    {"id": "quant_refresh",      "label": "Quant Dashboard",        "cadence": INTRADAY, "schedule": "every 3 min",              "category": "Quant",    "what": "Precomputes the Quant dashboard (~150 names) so the app never crunches on request."},
    {"id": "community_refresh",  "label": "Community Insights",     "cadence": INTRADAY, "schedule": "every 5 min",              "category": "Community","what": "Precomputes the community trending/verdict feed."},
    {"id": "watchlist_alerts",   "label": "Watchlist Alerts",       "cadence": INTRADAY, "schedule": "every 15 min",             "category": "Alerts",   "what": "Pushes watchlist state-change alerts."},
    {"id": "breakdown_alerts",   "label": "Breakdown Alerts",       "cadence": INTRADAY, "schedule": "every 15 min, RTH",         "category": "Signals",  "what": "Heavy-selling / breakdown scan — fires DISTRIB_FORMING + BREAKDOWN cards."},
    {"id": "breakout_alerts",    "label": "Breakout Alerts",        "cadence": INTRADAY, "schedule": "every 15 min, RTH",         "category": "Signals",  "what": "Unusual-buying / breakout scan — fires ACCUM_FORMING + BREAKOUT cards."},
    {"id": "regime_capture",     "label": "Regime Timeline",        "cadence": INTRADAY, "schedule": "every 5 min, 4a–8p ET",     "category": "Market",   "what": "Writes the market-regime timeline (on change) — the intraday regime history."},
    {"id": "cycle_signals",      "label": "Cycle Cards",            "cadence": INTRADAY, "schedule": "every 15 min, RTH",         "category": "Signals",  "what": "Turnaround (bottom) + Peak (top) tracked cards."},
    {"id": "social_snapshot",    "label": "Social Snapshot",        "cadence": INTRADAY, "schedule": "hourly",                   "category": "Community","what": "Snapshots social-trending mentions (the going-viral / track-record base)."},

    # ── session windows ──
    {"id": "premarket_alerts",   "label": "Pre-market Gap Alerts",  "cadence": SESSION,  "schedule": "every 15 min, 8:00–9:30a ET","category": "Alerts",  "what": "Disaster-gap alerts before the open."},
    {"id": "premarket_8am",      "label": "Pre-market Scan 8a",     "cadence": SESSION,  "schedule": "8:00a ET",                 "category": "Signals",  "what": "Pre-market opportunity scan."},
    {"id": "premarket_9am",      "label": "Pre-market Scan 9a",     "cadence": SESSION,  "schedule": "9:00a ET",                 "category": "Signals",  "what": "Pre-market opportunity scan."},

    # ── daily (post-close / EOD) ──
    {"id": "momentum_scan",      "label": "Momentum Scan",          "cadence": DAILY,    "schedule": "10:00a ET, Mon–Fri",       "category": "Signals",  "what": "Systematic cross-sectional momentum — fires TREND_MOMENTUM swings."},
    {"id": "bo_poc",             "label": "BO_POC Breakout POC",    "cadence": DAILY,    "schedule": "4:10p ET, Mon–Fri",        "category": "Signals",  "what": "Fidelity-matched confirmed-daily-close 20d-high breakout (validates the backtested archetype live)."},
    {"id": "drawdown_regime_log","label": "Drawdown-Regime Log",    "cadence": DAILY,    "schedule": "4:12p ET",                 "category": "Market",   "what": "Logs index % off 52-wk high (the deep-value accumulation window)."},
    {"id": "gate_validator",     "label": "Gate Validator",         "cadence": DAILY,    "schedule": "4:15p ET",                 "category": "Quant",    "what": "Judges rejected signals — would they have lost? (gate-correctness %)."},
    {"id": "deep_value_signal",  "label": "Deep-Value Combine",     "cadence": DAILY,    "schedule": "4:22p ET",                 "category": "Signals",  "what": "Fires the rare crash/deep-value buy when the regime + quality gates open."},
    {"id": "momentum_monitor",   "label": "Momentum Manager",       "cadence": DAILY,    "schedule": "4:25p ET, Mon–Fri",        "category": "Signals",  "what": "Manages TREND_MOMENTUM exits (chandelier trail + SMA50 backstop)."},
    {"id": "chart_read_log",     "label": "Chart-Read Track Record","cadence": DAILY,    "schedule": "4:40p ET, Mon–Fri",        "category": "Quant",    "what": "Records Expert-Read agreement vs realized move."},
    {"id": "phantom_audit",      "label": "Phantom-Data Audit",     "cadence": DAILY,    "schedule": "4:50p ET",                 "category": "Health",   "what": "Audits closed signals for bad-print / mismatch / out-of-range data integrity."},
    {"id": "analytics_report",   "label": "Analytics Report",       "cadence": DAILY,    "schedule": "5:30p ET",                 "category": "Quant",    "what": "Rolling-30d win-rate / R / drawdown report."},
    {"id": "daily_performance",  "label": "Daily Performance Snapshot","cadence": DAILY, "schedule": "8:05p ET",                 "category": "Quant",    "what": "Immutable per-day record: closed outcomes by detector/regime, give-back, active book, news catalysts."},
    {"id": "clear_zones_overnight","label": "Overnight Zone Clear",  "cadence": DAILY,   "schedule": "12:30a ET",                "category": "Health",   "what": "Clears armed zones overnight (after admin review window)."},
    {"id": "fundamentals_refresh","label": "Fundamentals Refresh",  "cadence": DAILY,    "schedule": "6:37a / 1:37p / 7:37p ET", "category": "Market",   "what": "Refreshes EDGAR XBRL fundamentals (rolling)."},

    # ── weekly ──
    {"id": "weight_optimization","label": "Weight Optimizer",       "cadence": WEEKLY,   "schedule": "Sun 2:00a UTC",            "category": "Quant",    "what": "Self-learning: re-optimizes L1–L9 scoring weights from real outcomes."},
]

_BY_ID = {j["id"]: j for j in JOBS}

CADENCE_ORDER = {INTRADAY: 0, SESSION: 1, DAILY: 2, WEEKLY: 3}


def merge(run_rows: list[dict] | None) -> list[dict]:
    """Merge the static catalog with the per-job last-run ledger rows.
    Returns one entry per known job (+ any unknown ids the ledger has seen),
    sorted by cadence then label. Pure."""
    runs = {r.get("job_id"): r for r in (run_rows or []) if r.get("job_id")}
    out: list[dict] = []

    def _entry(meta: dict, run: dict | None) -> dict:
        run = run or {}
        return {
            **{k: meta.get(k) for k in ("id", "label", "cadence", "schedule", "category", "what")},
            "last_run":   run.get("last_finished") or run.get("last_started"),
            "status":     run.get("last_status"),
            "duration_ms": run.get("last_duration_ms"),
            "summary":    run.get("last_summary"),
            "last_error": run.get("last_error"),
            "run_count":  run.get("run_count"),
            "error_count": run.get("error_count"),
        }

    for meta in JOBS:
        out.append(_entry(meta, runs.get(meta["id"])))
    # any ledger ids not in the catalog (new job not yet documented)
    for jid, run in runs.items():
        if jid not in _BY_ID:
            out.append(_entry({"id": jid, "label": jid, "cadence": DAILY,
                               "schedule": "—", "category": "Other",
                               "what": "(not yet documented)"}, run))

    out.sort(key=lambda e: (CADENCE_ORDER.get(e["cadence"], 9), e["label"]))
    return out
