"""
Phantom-data EOD audit.
=======================
Daily integrity check on CLOSED signals so a bad-price / phantom close can't
silently rot the track record for a month before we notice (the 2026-06-03
incident: fake stop-outs recorded at prices the tape never printed).

For every stock signal closed in the window it verifies the recorded exit
against that day's REAL 1-min range, plus cheap consistency checks:

  PHANTOM   — recorded exit price is OUTSIDE that day's 1-min [low, high]
              (the tape never printed it) → almost certainly a bad-print close.
  MISMATCH  — result='win' but result_pct<0, or result='loss' but >0.
  NO_PNL    — a stop_hit/target_hit close with no result_pct recorded.
  OVERSHOOT — (warn only) a stop_hit booked a loss bigger than the stop
              distance + 1% (suspicious even if inside the day's range).

Options are premium/delta-based (no clean tape comparison), so they get the
consistency checks only — the tape check is stock-only.

Run daily from runner (after the close) or on demand:
    python -m engine.phantom_audit --days=1
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("signalbolt.phantom_audit")

_TOL = 0.001   # 0.1% tolerance for rounding around the day's range


def _supabase():
    from supabase import create_client
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"],
    )


def audit(sb=None, days: int = 1) -> dict:
    """Audit signals closed in the last `days`. Returns a structured result."""
    sb = sb or _supabase()
    from engine import alpaca_client as ac

    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = (
        sb.table("signals")
          .select("id,ticker,direction,entry_price,stop_loss,result,result_pct,"
                  "closed_reason,closed_at,score_breakdown")
          .eq("status", "closed")
          .gte("closed_at", since)
          .order("closed_at")
          .limit(2000)
          .execute()
    ).data or []

    flagged: list[dict] = []
    audited = 0
    unverified = 0
    bars_cache: dict[str, object] = {}

    def day_range(ticker: str, date_utc):
        if ticker not in bars_cache:
            bars_cache[ticker] = ac.get_bars(ticker, "1Min", days=days + 2)
        df = bars_cache[ticker]
        if df is None or len(df) == 0:
            return None
        day = df[df.index.date == date_utc]
        if len(day) == 0:
            return None
        return float(day["low"].min()), float(day["high"].max())

    for r in rows:
        audited += 1
        ticker = r["ticker"]
        result = r.get("result")
        pct = r.get("result_pct")
        reason = r.get("closed_reason") or ""
        is_long = r["direction"] == "LONG"

        def _flag(kind, detail):
            flagged.append({
                "kind": kind, "id": r["id"], "ticker": ticker,
                "direction": r["direction"], "result": result, "result_pct": pct,
                "closed_reason": reason, "closed_at": r.get("closed_at"), "detail": detail,
            })

        # ── Consistency checks (no bars needed) ──────────────────────────────
        if reason in ("stop_hit", "target_hit") and pct is None:
            _flag("NO_PNL", "stop/target close with no result_pct recorded")
            continue
        if pct is not None:
            pct = float(pct)
            if result == "win" and pct < -0.05:
                _flag("MISMATCH", f"result=win but result_pct={pct:.2f}%")
            elif result == "loss" and pct > 0.05:
                _flag("MISMATCH", f"result=loss but result_pct={pct:.2f}%")

        # ── Tape check (needs price + bars) ──────────────────────────────────
        try:
            entry = float(r["entry_price"])
        except Exception:
            continue
        if pct is None or not entry:
            continue

        exit_px = entry * (1 + pct / 100) if is_long else entry * (1 - pct / 100)
        d = datetime.fromisoformat(r["closed_at"].replace("Z", "+00:00")).date()
        rng = day_range(ticker, d)
        if rng is None:
            unverified += 1
            continue
        lo, hi = rng
        if exit_px > hi * (1 + _TOL) or exit_px < lo * (1 - _TOL):
            _flag("PHANTOM",
                  f"recorded exit ~{exit_px:.2f} outside day range [{lo:.2f},{hi:.2f}] "
                  f"(reason={reason}) — tape never printed it")
            continue

        # Overshoot (warn): stop booked a loss bigger than the stop distance.
        try:
            stop = float(r["stop_loss"])
            stop_dist = abs(stop - entry) / entry * 100 if entry else 0
            if reason == "stop_hit" and pct < 0 and abs(pct) > stop_dist + 1.0:
                _flag("OVERSHOOT",
                      f"loss {pct:.1f}% exceeds stop distance {stop_dist:.1f}% (check fill)")
        except Exception:
            pass

    # de-dupe flags per id+kind
    seen, deduped = set(), []
    for f in flagged:
        k = (f["id"], f["kind"])
        if k not in seen:
            seen.add(k); deduped.append(f)

    serious = [f for f in deduped if f["kind"] in ("PHANTOM", "MISMATCH", "NO_PNL")]
    return {
        "since": since,
        "audited": audited,
        "unverified": unverified,
        "flagged_count": len(deduped),
        "serious_count": len(serious),
        "flagged": deduped,
    }


def run_and_alert(days: int = 1) -> dict:
    """Run the audit, log it, and push an ADMIN-ONLY summary (clean or flagged)."""
    res = audit(days=days)
    serious = res["serious_count"]
    n = res["audited"]
    if serious:
        logger.error(f"[phantom_audit] {serious} SERIOUS flag(s) of {n} audited: "
                     + "; ".join(f"{f['kind']} {f['ticker']} ({f['detail']})"
                                 for f in res["flagged"] if f["kind"] != "OVERSHOOT"))
        try:
            import sentry_sdk
            sentry_sdk.capture_message(
                f"phantom_audit: {serious} serious data-integrity flag(s) "
                f"of {n} closed signals", level="error")
        except Exception:
            pass
    else:
        logger.info(f"[phantom_audit] clean — {n} closed signals audited, "
                    f"{res['unverified']} unverified, 0 serious flags "
                    f"({res['flagged_count']} overshoot warnings)")

    try:
        from engine import push
        if serious:
            tickers = ", ".join(sorted({f["ticker"] for f in res["flagged"]
                                        if f["kind"] != "OVERSHOOT"})[:8])
            push.send_admin_alert(
                title=f"⚠️ EOD audit — {serious} data flag(s)",
                body=f"{serious} of {n} closed signals look wrong ({tickers}). Check /admin/phantom-audit.",
                data={"audit": "phantom", "serious": serious},
            )
        else:
            push.send_admin_alert(
                title=f"✅ EOD audit clean — {n} signals",
                body=f"All {n} closes verified against the tape. No phantom/mismatch data.",
                data={"audit": "phantom", "serious": 0},
            )
    except Exception as e:
        logger.warning(f"[phantom_audit] admin alert failed: {e}")
    return res


if __name__ == "__main__":
    import sys
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    _days = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--days=")), "1"))
    r = audit(days=_days)
    print(f"audited={r['audited']} unverified={r['unverified']} "
          f"flagged={r['flagged_count']} serious={r['serious_count']}")
    for f in r["flagged"]:
        print(f"  [{f['kind']}] {f['ticker']} {f['direction']} "
              f"result={f['result']} {f['result_pct']}% :: {f['detail']}")
