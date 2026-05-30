#!/usr/bin/env python3
"""
SignalBolt ops health check — meant to run every ~2h on market days.

Checks:
  1. Fly machines (engine + worker) are started & health-checks passing
  2. Recent Fly logs scanned for errors  +  endpoint latency (performance)
  3. Detectors active — signals firing + entry-gate activity in the last 24h
  4. Quant dashboard reachable + its 5-min lifecycle sync is running
  5. Armed-zone (detector-performance) episodes are being recorded

Prints a PASS/WARN/FAIL report. Exit code: 0 = ok, 1 = warnings, 2 = failures.
Run from the engine repo root:  python healthcheck.py
"""
import os
import sys
import time
import subprocess
from datetime import datetime, timezone, timedelta

# Make `engine` importable regardless of cwd.
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

ENGINE_URL = os.environ.get("HEALTHCHECK_ENGINE_URL", "https://signalbolt-engine.fly.dev")
ENGINE_APP = "signalbolt-engine"
WORKER_APP = "signalbolt-worker"

_results: list[tuple[str, str, str]] = []   # (level, name, detail)


def add(level: str, name: str, detail: str = "") -> None:
    _results.append((level, name, detail))


def _et_now() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now(timezone.utc)


def market_open_now() -> bool:
    et = _et_now()
    mins = et.hour * 60 + et.minute
    return et.weekday() < 5 and 570 <= mins <= 960   # 9:30 AM–4:00 PM ET


# ── 1. Fly machines ──────────────────────────────────────────────────────────
def check_fly(app: str) -> None:
    try:
        out = subprocess.run(["flyctl", "status", "-a", app],
                             capture_output=True, text=True, timeout=60)
        txt = (out.stdout or "") + (out.stderr or "")
        low = txt.lower()
        started = low.count("started")
        if "critical" in low or "failing" in low:
            add("FAIL", f"Fly {app}", "machine critical/failing")
        elif started == 0:
            add("FAIL", f"Fly {app}", "no started machines")
        elif "warning" in low:
            add("WARN", f"Fly {app}", f"{started} started, a check is warning")
        else:
            add("PASS", f"Fly {app}", f"{started} machine line(s) started, checks ok")
    except subprocess.TimeoutExpired:
        add("WARN", f"Fly {app}", "flyctl status timed out")
    except FileNotFoundError:
        add("INFO", f"Fly {app}", "flyctl not installed in this env")
    except Exception as e:
        add("WARN", f"Fly {app}", str(e)[:140])


# ── 2a. Recent log error scan ────────────────────────────────────────────────
_BENIGN = ("Alpaca API keys not set",)


def check_logs(app: str) -> None:
    try:
        out = subprocess.run(["flyctl", "logs", "-a", app, "--no-tail"],
                             capture_output=True, text=True, timeout=45)
        txt = out.stdout or ""
        if not txt:
            add("INFO", f"Logs {app}", "no log output (check Sentry/flyctl manually)")
            return
        lines = txt.splitlines()
        errs = [l for l in lines
                if any(k in l for k in ("ERROR", "CRITICAL", "Traceback", "Exception"))
                and not any(b in l for b in _BENIGN)]
        if errs:
            add("WARN", f"Logs {app}", f"{len(errs)} error line(s); latest: …{errs[-1][-140:]}")
        else:
            add("PASS", f"Logs {app}", f"no errors in {len(lines)} recent lines")
    except subprocess.TimeoutExpired:
        add("INFO", f"Logs {app}", "log scan timed out (skipped)")
    except FileNotFoundError:
        add("INFO", f"Logs {app}", "flyctl not installed")
    except Exception as e:
        add("INFO", f"Logs {app}", str(e)[:120])


# ── 2b. Endpoints + latency ──────────────────────────────────────────────────
def check_endpoints() -> None:
    try:
        import requests
    except Exception as e:
        add("WARN", "Endpoints", f"requests unavailable: {e}")
        return

    # /health (component checks) + latency
    try:
        t0 = time.monotonic()
        r = requests.get(ENGINE_URL + "/health", timeout=20)
        ms = round((time.monotonic() - t0) * 1000)
        body = (r.text or "").lower()
        if r.status_code != 200:
            add("FAIL", "/health", f"HTTP {r.status_code}")
        elif '"unhealthy"' in body or '"error"' in body or "false" in body and "degraded" in body:
            add("WARN", "/health", f"degraded component (200, {ms}ms)")
        else:
            level = "WARN" if ms > 4000 else "PASS"
            add(level, "/health", f"200 in {ms}ms")
    except Exception as e:
        add("FAIL", "/health", str(e)[:140])

    # Quant endpoints reachable (401 = up & auth-gated; 404/5xx = problem)
    for path in ("/quant/dashboard", "/quant/scorecard-all?days=30"):
        try:
            code = requests.get(ENGINE_URL + path, timeout=20).status_code
            if code in (200, 401):
                add("PASS", path.split("?")[0], f"reachable (HTTP {code})")
            else:
                add("FAIL", path.split("?")[0], f"HTTP {code}")
        except Exception as e:
            add("FAIL", path.split("?")[0], str(e)[:120])


# ── 3/4/5. Database activity (detectors, signals, quant sync, armed zones) ───
def check_db() -> None:
    try:
        from engine import config
        from supabase import create_client
        sb = create_client(config.SUPABASE_URL, config.SUPABASE_SECRET_KEY)
    except Exception as e:
        add("FAIL", "Supabase", f"connect failed: {str(e)[:120]}")
        return

    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    h24 = (now - timedelta(hours=24)).isoformat()
    open_mkt = market_open_now()

    # Signals firing
    try:
        rows = (sb.table("signals").select("id,created_at,status")
                  .gte("created_at", h24).order("created_at", desc=True)
                  .limit(300).execute().data) or []
        last = rows[0]["created_at"][:16] if rows else "—"
        lvl = "PASS" if rows else ("WARN" if open_mkt else "INFO")
        add(lvl, "Signals (24h)", f"{len(rows)} fired, last {last}")
    except Exception as e:
        add("WARN", "Signals", str(e)[:120])

    # Entry-gate / detector activity (rejections logged means the gate pipeline ran)
    try:
        gr = (sb.table("entry_gate_rejections").select("id", count="exact")
                .gte("created_at", h24).execute())
        add("PASS", "Entry gate (24h)", f"{gr.count} rejections logged")
    except Exception as e:
        add("INFO", "Entry gate", str(e)[:120])

    # Armed-zone episodes (detector-performance) recorded today
    try:
        az = (sb.table("armed_zone_history").select("id", count="exact")
                .gte("session_date", today).execute())
        lvl = "PASS" if az.count else ("WARN" if open_mkt else "INFO")
        add(lvl, "Armed zones (today)", f"{az.count} episodes")
    except Exception as e:
        add("WARN", "Armed zones", str(e)[:120])

    # Quant dashboard sync — the 5-min breakout/setup-watch lifecycle job
    try:
        bw = (sb.table("breakout_watch_history").select("last_seen_at")
                .order("last_seen_at", desc=True).limit(1).execute().data) or []
        if bw and bw[0].get("last_seen_at"):
            ls = datetime.fromisoformat(bw[0]["last_seen_at"].replace("Z", "+00:00"))
            age = int((now - ls).total_seconds() / 60)
            if open_mkt and age > 20:
                add("WARN", "Quant sync", f"last sync {age}m ago (>20m during RTH — job stalled?)")
            else:
                add("PASS", "Quant sync", f"last episode sync {age}m ago")
        else:
            add("WARN" if open_mkt else "INFO", "Quant sync", "no episodes recorded yet")
    except Exception as e:
        add("WARN", "Quant sync", str(e)[:120])


def main() -> None:
    et = _et_now()
    print(f"SignalBolt health check — {et:%Y-%m-%d %H:%M} ET   "
          f"(market {'OPEN' if market_open_now() else 'closed'})")
    print("=" * 72)

    check_fly(ENGINE_APP)
    check_fly(WORKER_APP)
    check_logs(ENGINE_APP)
    check_logs(WORKER_APP)
    check_endpoints()
    check_db()

    order = {"FAIL": 0, "WARN": 1, "PASS": 2, "INFO": 3}
    for lvl, name, detail in sorted(_results, key=lambda r: order.get(r[0], 9)):
        print(f"  [{lvl:4}] {name:22} {detail}")

    fails = sum(1 for r in _results if r[0] == "FAIL")
    warns = sum(1 for r in _results if r[0] == "WARN")
    passes = sum(1 for r in _results if r[0] == "PASS")
    print("=" * 72)
    print(f"SUMMARY: {fails} FAIL · {warns} WARN · {passes} PASS")
    sys.exit(2 if fails else (1 if warns else 0))


if __name__ == "__main__":
    main()
