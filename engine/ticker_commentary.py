"""
Ticker Commentary — a "play-by-play" of intraday technical EVENTS for one ticker.

This is the MVP of the live-commentary feature: open a ticker → see a day-long,
chronological feed of *transitions* (not a per-tick narration): MACD crosses,
EMA9/21 crosses, RSI overbought/oversold, VWAP reclaim/lose, opening-range
breaks, volume spikes, sharp moves (jump/dump), new highs/lows of day, the
opening gap — on 5m and 15m — plus occasional educational intraday ideas.

Design:
  • STATELESS REPLAY — `build(symbol)` re-derives the whole day's events from
    today's intraday bars each call (no scheduler / no state table for the MVP;
    V2 adds a background job + push, V3 ranking + a track record).
  • NOISE CONTROL — events fire only on a *transition*, with a per-type cooldown
    and a daily cap, ranked by severity. A flat tape produces (almost) nothing.
  • HONEST — ideas use educational framing ("setup favors", "invalidation below"),
    never "buy now"/"guaranteed". Intraday technical events are awareness, not edge.
  • DEFENSIVE — never raises into the request path; short/empty data → unavailable.

The swing/daily idea is intentionally NOT recomputed here — the ticker hub already
loads chart_read + decision_support, so the UI pairs this intraday feed with that
swing read.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger("signalbolt.ticker_commentary")

_DISCLAIMER = "Live technical commentary — educational awareness, not financial advice."
_MAX_EVENTS = 30
_MOVE_PCT = 1.2          # single-bar % move to flag a surge/dump
_VOL_SPIKE = 3.0         # bar volume vs trailing avg to flag a volume spike

# per-(type) cooldown in BARS on the detection timeframe (avoids machine-gun events)
_COOLDOWN = {
    "MACD_CROSS": 6, "EMA_CROSS": 6, "RSI": 8, "VWAP": 6, "ORB": 9999,
    "VOLUME": 4, "MOVE": 2, "HOD": 10, "LOD": 10, "LEVEL": 8, "GAP": 9999, "IDEA": 12,
}


# ── indicators (inline, self-contained) ───────────────────────────────────────
def _ema(s: pd.Series, span: int) -> np.ndarray:
    return s.ewm(span=span, adjust=False).mean().values


def _macd(close: pd.Series):
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.values, signal.values, (macd - signal).values


def _rsi(close: pd.Series, period: int = 14) -> np.ndarray:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # zero-loss windows: full strength if there were gains, neutral if perfectly flat
    rsi = rsi.mask(roll_dn == 0, np.where(roll_up > 0, 100.0, 50.0))
    return rsi.fillna(50.0).values


def _vwap(df: pd.DataFrame) -> np.ndarray:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_v = df["volume"].cumsum().replace(0, np.nan)
    return (tp * df["volume"]).cumsum().div(cum_v).fillna(df["close"]).values


def _session_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Rows belonging to the latest calendar (UTC) day present — i.e. today's session."""
    if df is None or df.empty:
        return df
    last_day = df.index[-1].date()
    return df[df.index.map(lambda t: t.date() == last_day)]


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df.resample(rule, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return o.dropna()


def _round(v) -> float:
    return round(float(v), 2)


# ── educational intraday idea (no advice language) ────────────────────────────
def _intraday_idea(tone: str, price: float, swing_lo: float, swing_hi: float, atr: float) -> dict | None:
    if not price or not atr:
        return None
    if tone == "bullish":
        stop = _round(min(swing_lo, price - 1.2 * atr))
        tgt = _round(max(swing_hi, price + 1.6 * atr))
        if price <= stop:
            return None
        rr = (tgt - price) / (price - stop) if price > stop else 0
        return {"bias": "long", "entry": _round(price), "invalidation": stop, "target": tgt,
                "rr": round(rr, 1),
                "text": f"Intraday: setup favors a long near ${_round(price)} — invalidation below "
                        f"${stop}, first upside ~${tgt} (R:R {round(rr,1)}). Educational, not advice."}
    if tone == "bearish":
        stop = _round(max(swing_hi, price + 1.2 * atr))
        tgt = _round(min(swing_lo, price - 1.6 * atr))
        if price >= stop:
            return None
        rr = (price - tgt) / (stop - price) if stop > price else 0
        return {"bias": "short", "entry": _round(price), "invalidation": stop, "target": tgt,
                "rr": round(rr, 1),
                "text": f"Intraday: setup favors a short near ${_round(price)} — invalidation above "
                        f"${stop}, first downside ~${tgt} (R:R {round(rr,1)}). Educational, not advice."}
    return None


# ── event detection (pure walk over one timeframe) ────────────────────────────
def _detect_tf(df: pd.DataFrame, tf_label: str, prior_close: float | None,
               want_ideas: bool) -> list[dict]:
    """Walk the bars of one timeframe and emit transition events. Pure."""
    n = len(df)
    if n < 16:
        return []
    close = df["close"]
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    vol = df["volume"].values.astype(float)
    px = close.values.astype(float)
    idx = df.index

    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    macd, sig, hist = _macd(close)
    rsi = _rsi(close)
    vwap = _vwap(df)
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - px[:-1]), abs(low[1:] - px[:-1])))

    events: list[dict] = []
    last_emit: dict[str, int] = {}

    def ok(kind: str, i: int) -> bool:
        return (i - last_emit.get(kind, -10_000)) >= _COOLDOWN.get(kind, 4)

    def emit(kind, i, tone, sev, title, detail, price, idea=None):
        last_emit[kind] = i
        ev = {"time": idx[i].isoformat(), "tf": tf_label, "type": kind, "tone": tone,
              "severity": sev, "title": title, "detail": detail, "price": _round(price)}
        if idea:
            ev["idea"] = idea
        events.append(ev)

    # opening gap (first bar of the session vs prior session close)
    if prior_close and prior_close > 0:
        gp = (px[0] - prior_close) / prior_close * 100
        if abs(gp) >= 1.0:
            up = gp > 0
            emit("GAP", 0, "bullish" if up else "bearish", 2,
                 f"Gapped {'up' if up else 'down'} {abs(gp):.1f}% at the open",
                 f"Opened at ${_round(px[0])} vs prior close ${_round(prior_close)} "
                 f"({'+' if up else ''}{gp:.1f}%).", px[0])

    # opening range (first 6 bars) → break events (once each)
    orb_hi = float(np.max(high[:6])) if n >= 6 else None
    orb_lo = float(np.min(low[:6])) if n >= 6 else None
    orb_done = {"up": False, "down": False}

    sess_hi = float(high[0]); sess_lo = float(low[0])

    warm = 6
    for i in range(warm, n):
        atr = float(np.mean(tr[max(0, i - 14):i])) if i > 1 else 0.0

        # MACD histogram sign flip = MACD/signal cross
        if ok("MACD_CROSS", i) and hist[i - 1] is not None:
            if hist[i - 1] <= 0 < hist[i]:
                idea = _intraday_idea("bullish", px[i], float(np.min(low[max(0, i - 6):i + 1])), sess_hi, atr) if want_ideas and ok("IDEA", i) else None
                if idea: last_emit["IDEA"] = i
                emit("MACD_CROSS", i, "bullish", 3, f"MACD bullish crossover ({tf_label})",
                     f"MACD crossed above its signal at ${_round(px[i])} — momentum turning up.", px[i], idea)
            elif hist[i - 1] >= 0 > hist[i]:
                idea = _intraday_idea("bearish", px[i], sess_lo, float(np.max(high[max(0, i - 6):i + 1])), atr) if want_ideas and ok("IDEA", i) else None
                if idea: last_emit["IDEA"] = i
                emit("MACD_CROSS", i, "bearish", 3, f"MACD bearish crossover ({tf_label})",
                     f"MACD crossed below its signal at ${_round(px[i])} — momentum turning down.", px[i], idea)

        # EMA 9/21 cross
        if ok("EMA_CROSS", i):
            if ema9[i - 1] <= ema21[i - 1] and ema9[i] > ema21[i]:
                emit("EMA_CROSS", i, "bullish", 2, f"9/21 EMA bullish cross ({tf_label})",
                     f"The 9 EMA crossed above the 21 EMA near ${_round(px[i])} — short-term trend turning up.", px[i])
            elif ema9[i - 1] >= ema21[i - 1] and ema9[i] < ema21[i]:
                emit("EMA_CROSS", i, "bearish", 2, f"9/21 EMA bearish cross ({tf_label})",
                     f"The 9 EMA crossed below the 21 EMA near ${_round(px[i])} — short-term trend turning down.", px[i])

        # RSI overbought / oversold (entering)
        if ok("RSI", i):
            if rsi[i - 1] < 70 <= rsi[i]:
                emit("RSI", i, "bearish", 1, f"RSI overbought ({tf_label})",
                     f"RSI pushed above 70 ({rsi[i]:.0f}) — stretched; momentum strong but extended.", px[i])
            elif rsi[i - 1] > 30 >= rsi[i]:
                emit("RSI", i, "bullish", 1, f"RSI oversold ({tf_label})",
                     f"RSI dropped below 30 ({rsi[i]:.0f}) — washed out; watch for a bounce.", px[i])

        # VWAP reclaim / lose
        if ok("VWAP", i):
            if px[i - 1] < vwap[i - 1] and px[i] > vwap[i]:
                emit("VWAP", i, "bullish", 2, f"Reclaimed VWAP ({tf_label})",
                     f"Price reclaimed VWAP (${_round(vwap[i])}) — buyers back in control intraday.", px[i])
            elif px[i - 1] > vwap[i - 1] and px[i] < vwap[i]:
                emit("VWAP", i, "bearish", 2, f"Lost VWAP ({tf_label})",
                     f"Price lost VWAP (${_round(vwap[i])}) — sellers in control intraday.", px[i])

        # opening-range break (once each direction)
        if orb_hi and not orb_done["up"] and px[i] > orb_hi:
            orb_done["up"] = True
            emit("ORB", i, "bullish", 2, f"Broke the opening range high ({tf_label})",
                 f"Cleared the first-30m high ${_round(orb_hi)} — intraday breakout attempt.", px[i])
        if orb_lo and not orb_done["down"] and px[i] < orb_lo:
            orb_done["down"] = True
            emit("ORB", i, "bearish", 2, f"Broke the opening range low ({tf_label})",
                 f"Lost the first-30m low ${_round(orb_lo)} — intraday breakdown attempt.", px[i])

        # volume spike
        avg_v = float(np.mean(vol[max(0, i - 20):i])) or 1.0
        if ok("VOLUME", i) and vol[i] >= _VOL_SPIKE * avg_v:
            up = px[i] >= px[i - 1]
            emit("VOLUME", i, "bullish" if up else "bearish", 2, f"Volume spike ({tf_label})",
                 f"Volume hit ~{vol[i] / avg_v:.1f}× the recent average on a {'green' if up else 'red'} bar.", px[i])

        # sharp single-bar move
        if ok("MOVE", i) and px[i - 1] > 0:
            chg = (px[i] - px[i - 1]) / px[i - 1] * 100
            if chg >= _MOVE_PCT:
                emit("MOVE", i, "bullish", 2, f"Sharp jump ({tf_label})",
                     f"Popped +{chg:.1f}% in one bar to ${_round(px[i])}.", px[i])
            elif chg <= -_MOVE_PCT:
                emit("MOVE", i, "bearish", 2, f"Sharp drop ({tf_label})",
                     f"Dropped {chg:.1f}% in one bar to ${_round(px[i])}.", px[i])

        # new high / low of day
        if high[i] > sess_hi:
            sess_hi = float(high[i])
            if ok("HOD", i) and i > 8:
                emit("HOD", i, "bullish", 1, f"New high of day ({tf_label})",
                     f"Tagged a fresh session high at ${_round(high[i])}.", high[i])
        if low[i] < sess_lo:
            sess_lo = float(low[i])
            if ok("LOD", i) and i > 8:
                emit("LOD", i, "bearish", 1, f"New low of day ({tf_label})",
                     f"Printed a fresh session low at ${_round(low[i])}.", low[i])

    return events


# ── public ─────────────────────────────────────────────────────────────────────
def _unavailable(symbol: str, note: str) -> dict:
    return {"available": False, "ticker": symbol, "note": note, "disclaimer": _DISCLAIMER}


def build(symbol: str, now: datetime | None = None) -> dict:
    """Build today's intraday commentary feed for `symbol`. Never raises."""
    sym = (symbol or "").upper().strip()
    if not sym:
        return _unavailable(sym, "No ticker.")
    try:
        from engine.alpaca_client import get_bars
        raw = get_bars(sym, "5Min", days=4)
        if raw is None or raw.empty:
            return _unavailable(sym, "No intraday data yet.")
        # prior session's close (for the opening-gap event)
        last_day = raw.index[-1].date()
        prior = raw[raw.index.map(lambda t: t.date() != last_day)]
        prior_close = float(prior["close"].iloc[-1]) if len(prior) else None

        df5 = _session_slice(raw)
        if df5 is None or len(df5) < 16:
            return _unavailable(sym, "Not enough bars in the current session yet — check back after the open.")

        events = _detect_tf(df5, "5m", prior_close, want_ideas=True)
        try:
            df15 = _resample(df5, "15min")
            events += _detect_tf(df15, "15m", None, want_ideas=False)
        except Exception:
            pass

        # newest first, cap, and pull the most recent idea forward as the "current" idea
        events.sort(key=lambda e: e["time"], reverse=True)
        current_idea = next((e["idea"] for e in events if e.get("idea")), None)
        events = events[:_MAX_EVENTS]

        last_px = _round(df5["close"].iloc[-1])
        bull = sum(1 for e in events if e["tone"] == "bullish")
        bear = sum(1 for e in events if e["tone"] == "bearish")
        lean = "bullish" if bull > bear + 1 else "bearish" if bear > bull + 1 else "mixed"
        summary = (f"{len(events)} event(s) today — intraday tape leans {lean} "
                   f"({bull} bullish / {bear} bearish)." if events
                   else "Quiet tape so far — no notable intraday technical events yet.")

        return {
            "available": True, "ticker": sym,
            "session_date": str(last_day),
            "as_of": (now or datetime.now(timezone.utc)).isoformat(),
            "price": last_px, "lean": lean, "summary": summary,
            "current_idea": current_idea, "events": events,
            "disclaimer": _DISCLAIMER,
        }
    except Exception as e:
        logger.debug(f"[ticker_commentary] build {sym} failed: {e}")
        return _unavailable(sym, "Commentary unavailable.")
