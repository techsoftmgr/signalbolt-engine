"""
Market-wide movers for the Movers tab — the biggest % gainers / losers and the
most unusual-volume names across a broad LIQUID universe, filtered to real common
stocks (no warrants / units / rights / penny pumps) and overlaid with our quant
read + any active signal.

Candidate pool = our 245-name liquid universe (prescreener.EXTENDED_UNIVERSE) +
momentum universe + Alpaca's market-wide most-actives & screener feeds for breadth.
Everything is re-priced from one batched daily-bars call so % change / volume /
relative-volume are consistent and a liquidity floor can be applied. 60s cached.
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

logger = logging.getLogger("signalbolt.movers")

_CACHE_KEY      = "markets:movers:v1"
_TTL            = 180   # outlive the 120s warmer interval so a request never hits an empty cache
_inflight       = threading.Lock()   # coalesce concurrent heavy builds (~15-25s) to one
_MIN_PRICE      = float(os.environ.get("MOVERS_MIN_PRICE", "5"))            # drop sub-$5 penny names
_MIN_DOLLARVOL  = float(os.environ.get("MOVERS_MIN_DOLLARVOL", "5000000"))  # $5M floor: just drops dead/halted names. Pumps are already killed by market-cap vetting (screener) + curated-only construction, so this no longer needs to be high.
_UNUSUAL_RELVOL = float(os.environ.get("MOVERS_UNUSUAL_RELVOL", "2.0"))     # 2× the 20-day avg = unusual
_MIN_MKTCAP     = float(os.environ.get("MOVERS_MIN_MKTCAP", "1500000000"))  # $1.5B floor to vet screener names


def _is_common(s: str) -> bool:
    """Real common-stock ticker heuristic: pure A-Z, ≤5 chars, and not a 5-letter
    symbol ending in W/U/R/Q (warrant / unit / right / bankruptcy). Drops the
    GRAF.WS / IVDAW / HSPTU junk the raw screener is full of."""
    return bool(s) and s.isalpha() and s.isupper() and len(s) <= 5 and not (len(s) == 5 and s[-1] in "WURQ")


def _screener(path: str, top: int = 50) -> dict:
    """Alpaca market-wide screener (keyed by our existing Alpaca creds). top caps at 50."""
    try:
        import httpx
        key = os.environ.get("ALPACA_API_KEY")
        sec = os.environ.get("ALPACA_SECRET_KEY")
        if not key or not sec:
            return {}
        with httpx.Client(timeout=12, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}) as c:
            r = c.get(f"https://data.alpaca.markets/v1beta1/screener/stocks/{path}", params={"top": top})
            return r.json() if r.status_code == 200 else {}
    except Exception as e:
        logger.debug(f"[movers] screener {path} failed: {e}")
        return {}


def _vetted_screener_symbols() -> list[str]:
    """Alpaca's market-wide top-50 gainers/losers, but only the ones that survive a
    market-cap floor — the one reliable filter that separates a real mover (ROKU,
    MAAS) from a microcap pump-and-dump (UBXG +67%, DSY +33%). Market caps are
    24h-cached, so only the first refresh of the day pays the lookups."""
    cand: list[str] = []
    mv = _screener("movers", 50)
    for it in (mv.get("gainers") or []) + (mv.get("losers") or []):
        s = it.get("symbol")
        if s and _is_common(s) and (it.get("price") or 0) >= _MIN_PRICE:
            cand.append(s)
    if not cand:
        return []
    # Market-cap lookups are slow (Nasdaq + Polygon per symbol) — fetch them in
    # PARALLEL so vetting ~30 names takes a few seconds, not ~100s sequentially.
    # (Each tf.get is 24h-cached, so warm runs are instant regardless.)
    def _vet(s: str):
        try:
            from engine import ticker_fundamentals as tf
            mc = (tf.get(s) or {}).get("market_cap")
            return s if (mc and mc >= _MIN_MKTCAP) else None
        except Exception:
            return None
    try:
        with ThreadPoolExecutor(max_workers=10) as ex:
            return [r for r in ex.map(_vet, cand) if r]
    except Exception as e:
        logger.debug(f"[movers] screener vetting failed: {e}")
        return []


def _candidate_symbols() -> list[str]:
    """Hybrid pool: our curated broad-LIQUID universe (~250 names incl.
    RIVN/ROKU/ARM/INTC) ∪ Alpaca's market-wide top-50 movers vetted by market cap.
    The curated set guarantees clean, fast coverage of tracked names; the vetted
    screener adds true market-wide breadth (catches a real mover outside our list,
    e.g. MAAS) WITHOUT the warrant/penny-pump junk the raw screener is full of."""
    syms: set[str] = set()
    try:
        from engine import prescreener as ps, momentum_detector as md
        syms |= set(ps.EXTENDED_UNIVERSE) | set(md.UNIVERSE)
    except Exception as e:
        logger.debug(f"[movers] universe load failed: {e}")
    syms |= set(_vetted_screener_symbols())
    return [s for s in syms if _is_common(s)]


def peek_movers() -> dict | None:
    """Fast, non-blocking read of the cached movers (None if not warmed yet).
    The endpoint uses this so a user request NEVER triggers the ~15-25s build —
    that's the warmer's job (runner._run_warm_movers)."""
    try:
        from engine import cache
        return cache.kv.get_json(_CACHE_KEY)
    except Exception:
        return None


def compute_movers(limit: int = 20, force: bool = False) -> dict:
    """{asOf, gainers[], losers[], unusualVolume[]} — each item: symbol, price,
    changePct, volume, relVol, rsi?, setupType?, signal?. Cached; the heavy build
    (market-cap vetting + bars) is coalesced to one in-flight worker via a lock."""
    from engine import cache
    empty = {"asOf": datetime.now(timezone.utc).isoformat(), "gainers": [], "losers": [], "unusualVolume": []}
    if not force:
        cached = cache.kv.get_json(_CACHE_KEY)
        if cached:
            return cached
    # Only ONE heavy build at a time — concurrent callers get whatever's cached.
    if not _inflight.acquire(blocking=False):
        return cache.kv.get_json(_CACHE_KEY) or empty
    try:
        cached = cache.kv.get_json(_CACHE_KEY)
        if cached and not force:        # filled while we waited for the lock
            return cached

        syms = _candidate_symbols()
        if not syms:
            return empty

        from engine.alpaca_client import get_multi_bars
        bars = get_multi_bars(syms, "1Day", 40) or {}
        if not bars:
            return empty

        # Quant overlay (RSI / setup) from the cached scan — present only for tracked names.
        quant: dict = {}
        try:
            for r in (cache.kv.get_json("quant:scored:v1") or []):
                if r.get("ticker"):
                    quant[r["ticker"]] = r
        except Exception:
            pass

        # Active-signal overlay (so a mover we already have a trade on is flagged).
        sig: dict = {}
        try:
            from engine import runner
            sb = runner._supabase()
            present = list(bars.keys())
            for i in range(0, len(present), 200):
                rows = (sb.table("signals").select("ticker,direction")
                        .eq("status", "active").in_("ticker", present[i:i + 200]).execute().data) or []
                for r in rows:
                    sig[r["ticker"]] = r.get("direction")
        except Exception as e:
            logger.debug(f"[movers] active-signal overlay failed: {e}")

        built: list[dict] = []
        for s, df in bars.items():
            try:
                if df is None or len(df) < 2:
                    continue
                price = float(df["close"].iloc[-1])
                prev  = float(df["close"].iloc[-2])
                vol   = float(df["volume"].iloc[-1])
                if price < _MIN_PRICE or prev <= 0 or price * vol < _MIN_DOLLARVOL:
                    continue
                chg = round((price - prev) / prev * 100, 2)
                avg = float(df["volume"].iloc[-21:-1].mean()) if len(df) >= 21 else float(df["volume"].iloc[:-1].mean() or 0)
                relvol = round(vol / avg, 2) if avg > 0 else None
                q = quant.get(s) or {}
                built.append({
                    "symbol": s, "price": round(price, 2), "changePct": chg, "volume": int(vol),
                    "relVol": relvol, "rsi": q.get("rsi"), "setupType": q.get("setupType"), "signal": sig.get(s),
                })
            except Exception:
                continue

        gainers = sorted([b for b in built if b["changePct"] > 0], key=lambda x: -x["changePct"])[:limit]
        losers  = sorted([b for b in built if b["changePct"] < 0], key=lambda x:  x["changePct"])[:limit]
        unusual = sorted([b for b in built if (b["relVol"] or 0) >= _UNUSUAL_RELVOL],
                         key=lambda x: -(x["relVol"] or 0))[:limit]

        out = {"asOf": datetime.now(timezone.utc).isoformat(),
               "gainers": gainers, "losers": losers, "unusualVolume": unusual}
        try:
            cache.kv.set_json(_CACHE_KEY, out, _TTL)
        except Exception:
            pass
        return out
    finally:
        _inflight.release()
