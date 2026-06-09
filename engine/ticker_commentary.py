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

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _ET = None

logger = logging.getLogger("signalbolt.ticker_commentary")

_DISCLAIMER = "Live technical commentary for educational awareness. Not financial advice."
_MAX_EVENTS = 30
_MOVE_PCT = 1.2          # single-bar % move to flag a surge/dump (regular hours)
_MOVE_PCT_EXT = 2.5      # bigger move required pre/after-hours (thin tape)
_VOL_SPIKE = 3.0         # bar volume vs trailing avg (regular hours)
_VOL_SPIKE_EXT = 5.0     # higher bar pre/after-hours so a thin print isn't a "spike"


def _bar_session(ts) -> str:
    """'pre' / 'rth' / 'after' from a (UTC) bar timestamp, evaluated in ET."""
    try:
        et = ts.tz_convert(_ET) if hasattr(ts, "tz_convert") else ts.astimezone(_ET)
        mins = et.hour * 60 + et.minute
        if mins < 9 * 60 + 30:
            return "pre"
        if mins >= 16 * 60:
            return "after"
        return "rth"
    except Exception:
        return "rth"

# per-(type) cooldown in BARS on the detection timeframe (avoids machine-gun events)
_COOLDOWN = {
    "MACD_CROSS": 6, "EMA_CROSS": 6, "RSI": 8, "VWAP": 6, "ORB": 9999,
    "VOLUME": 4, "MOVE": 2, "HOD": 10, "LOD": 10, "LEVEL": 8, "GAP": 9999, "IDEA": 6,
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
# Stops sit a buffer BEYOND the recent swing (not right at it) so a normal wick
# doesn't trigger them, and an idea is only returned when R:R clears _MIN_RR —
# a sub-1 reward-to-risk "idea" is no idea at all.
_MIN_RR = 1.2
_STOP_BUFFER_ATR = 0.3


def _intraday_idea(tone: str, price: float, swing_lo: float, swing_hi: float, atr: float,
                   min_rr: float = _MIN_RR) -> dict | None:
    if not price or not atr:
        return None
    buf = _STOP_BUFFER_ATR * atr
    if tone == "bullish":
        stop = _round(min(swing_lo, price - 1.2 * atr) - buf)   # below the swing low + buffer
        tgt = _round(max(swing_hi, price + 1.6 * atr))
        if price <= stop:
            return None
        rr = (tgt - price) / (price - stop)
        if rr < min_rr:
            return None
        return {"bias": "long", "entry": _round(price), "invalidation": stop, "target": tgt,
                "rr": round(rr, 1),
                "text": f"Intraday long plan (if/then). If it holds near ${_round(price)}, invalidation is below "
                        f"${stop} and the first level is ~${tgt} (R:R {round(rr,1)}). Educational, not a prediction."}
    if tone == "bearish":
        stop = _round(max(swing_hi, price + 1.2 * atr) + buf)   # above the swing high + buffer
        tgt = _round(min(swing_lo, price - 1.6 * atr))
        if price >= stop:
            return None
        rr = (price - tgt) / (stop - price)
        if rr < min_rr:
            return None
        return {"bias": "short", "entry": _round(price), "invalidation": stop, "target": tgt,
                "rr": round(rr, 1),
                "text": f"Intraday short plan (if/then). If it rejects near ${_round(price)}, invalidation is above "
                        f"${stop} and the first level is ~${tgt} (R:R {round(rr,1)}). Educational, not a prediction."}
    return None


def _session_bias(df5: pd.DataFrame, df15: pd.DataFrame | None) -> str:
    """The day's dominant direction from price-vs-VWAP + the 15m EMA9/21.
    'up' / 'down' require BOTH to agree; otherwise 'neutral'. Ideas only fire WITH
    this bias; counter-trend events are flagged 'watch only'. Per-ticker intraday
    only — does NOT use the market regime detector."""
    try:
        price = float(df5["close"].iloc[-1])
        above_vwap = price > _vwap(df5)[-1]
        if df15 is not None and len(df15) >= 21:
            e9 = _ema(df15["close"], 9)[-1]
            e21 = _ema(df15["close"], 21)[-1]
            if above_vwap and e9 > e21:
                return "up"
            if (not above_vwap) and e9 < e21:
                return "down"
            return "neutral"
        # 15m not available yet → VWAP-only lean
        return "up" if above_vwap else "down"
    except Exception:
        return "neutral"


# ── event detection (pure walk over one timeframe) ────────────────────────────
def _detect_tf(df: pd.DataFrame, tf_label: str, prior_close: float | None,
               want_ideas: bool, bias: str = "neutral") -> list[dict]:
    """Walk the bars of one timeframe and emit transition events. Pure.

    `bias` ('up'/'down'/'neutral') is the session direction: ideas attach ONLY to
    events that agree with it, and events that oppose it are flagged
    `against_trend` (and never carry an idea) so the feed doesn't read as a
    direction flip on every counter-trend wiggle."""
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

    def _against(tone: str) -> bool:
        return (tone == "bullish" and bias == "down") or (tone == "bearish" and bias == "up")

    def emit(kind, i, tone, sev, title, detail, price, idea=None):
        last_emit[kind] = i
        against = _against(tone)
        if against:
            detail = f"{detail} Counter-trend: the tape is {bias}, so this is lower-odds. Watch only."
        ev = {"time": idx[i].isoformat(), "tf": tf_label, "type": kind, "tone": tone,
              "severity": sev, "title": title, "detail": detail, "price": _round(price),
              "against_trend": against, "session": _bar_session(idx[i])}
        # An idea is attached ONLY when it agrees with the session bias — never
        # against the tape (that's what produced the losing counter-trend long).
        if idea and not against:
            ev["idea"] = idea
        events.append(ev)

    def maybe_idea(tone: str, i: int, atr: float):
        """Build a with-trend idea for a continuation trigger (MACD / EMA / VWAP),
        bias-aligned + R:R-gated + IDEA-cooldown'd. Returns None otherwise."""
        if not (want_ideas and ok("IDEA", i)):
            return None
        aligned = (tone == "bullish" and bias == "up") or (tone == "bearish" and bias == "down")
        if not aligned:
            return None
        if tone == "bullish":
            idea = _intraday_idea("bullish", px[i], float(np.min(low[max(0, i - 6):i + 1])), sess_hi, atr)
        else:
            idea = _intraday_idea("bearish", px[i], sess_lo, float(np.max(high[max(0, i - 6):i + 1])), atr)
        if idea:
            last_emit["IDEA"] = i
        return idea

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

        # MACD histogram sign flip = MACD/signal cross. Ideas only when the cross
        # AGREES with the session bias (no counter-trend ideas).
        if ok("MACD_CROSS", i) and hist[i - 1] is not None:
            if hist[i - 1] <= 0 < hist[i]:
                emit("MACD_CROSS", i, "bullish", 3, f"MACD bullish crossover ({tf_label})",
                     f"MACD crossed above its signal at ${_round(px[i])}; momentum is turning up.",
                     px[i], maybe_idea("bullish", i, atr))
            elif hist[i - 1] >= 0 > hist[i]:
                emit("MACD_CROSS", i, "bearish", 3, f"MACD bearish crossover ({tf_label})",
                     f"MACD crossed below its signal at ${_round(px[i])}; momentum is turning down.",
                     px[i], maybe_idea("bearish", i, atr))

        # EMA 9/21 cross — a with-trend cross is a continuation entry, so it can
        # carry an idea (fade-the-bounce in a downtrend, buy-the-dip in an uptrend).
        if ok("EMA_CROSS", i):
            if ema9[i - 1] <= ema21[i - 1] and ema9[i] > ema21[i]:
                emit("EMA_CROSS", i, "bullish", 2, f"9/21 EMA bullish cross ({tf_label})",
                     f"The 9 EMA crossed above the 21 EMA near ${_round(px[i])}; short-term trend is turning up.",
                     px[i], maybe_idea("bullish", i, atr))
            elif ema9[i - 1] >= ema21[i - 1] and ema9[i] < ema21[i]:
                emit("EMA_CROSS", i, "bearish", 2, f"9/21 EMA bearish cross ({tf_label})",
                     f"The 9 EMA crossed below the 21 EMA near ${_round(px[i])}; short-term trend is turning down.",
                     px[i], maybe_idea("bearish", i, atr))

        # RSI overbought / oversold (entering)
        if ok("RSI", i):
            if rsi[i - 1] < 70 <= rsi[i]:
                emit("RSI", i, "bearish", 1, f"RSI overbought ({tf_label})",
                     f"RSI pushed above 70 ({rsi[i]:.0f}). Momentum is strong but stretched.", px[i])
            elif rsi[i - 1] > 30 >= rsi[i]:
                emit("RSI", i, "bullish", 1, f"RSI oversold ({tf_label})",
                     f"RSI dropped below 30 ({rsi[i]:.0f}). Washed out, so watch for a bounce.", px[i])

        # VWAP reclaim / lose — a classic with-trend continuation trigger, so it
        # can carry an idea when it agrees with the session bias.
        if ok("VWAP", i):
            if px[i - 1] < vwap[i - 1] and px[i] > vwap[i]:
                emit("VWAP", i, "bullish", 2, f"Reclaimed VWAP ({tf_label})",
                     f"Price reclaimed VWAP (${_round(vwap[i])}); buyers are back in control intraday.",
                     px[i], maybe_idea("bullish", i, atr))
            elif px[i - 1] > vwap[i - 1] and px[i] < vwap[i]:
                emit("VWAP", i, "bearish", 2, f"Lost VWAP ({tf_label})",
                     f"Price lost VWAP (${_round(vwap[i])}); sellers are in control intraday.",
                     px[i], maybe_idea("bearish", i, atr))

        # opening-range break (once each direction)
        if orb_hi and not orb_done["up"] and px[i] > orb_hi:
            orb_done["up"] = True
            emit("ORB", i, "bullish", 2, f"Broke the opening range high ({tf_label})",
                 f"Cleared the first-30m high ${_round(orb_hi)}; an intraday breakout attempt.", px[i])
        if orb_lo and not orb_done["down"] and px[i] < orb_lo:
            orb_done["down"] = True
            emit("ORB", i, "bearish", 2, f"Broke the opening range low ({tf_label})",
                 f"Lost the first-30m low ${_round(orb_lo)}; an intraday breakdown attempt.", px[i])

        # session-aware thresholds — pre/after-hours tape is thin, so a "spike" or
        # "sharp move" needs to clear a higher bar before it's worth surfacing.
        sess_i = _bar_session(idx[i])
        ext = sess_i in ("pre", "after")
        sess_word = {"pre": "Premarket ", "after": "After-hours "}.get(sess_i, "")
        thin = " (thin premarket tape)" if sess_i == "pre" else " (thin after-hours tape)" if sess_i == "after" else ""

        # volume spike
        avg_v = float(np.mean(vol[max(0, i - 20):i])) or 1.0
        vol_mult = _VOL_SPIKE_EXT if ext else _VOL_SPIKE
        if ok("VOLUME", i) and vol[i] >= vol_mult * avg_v:
            up = px[i] >= px[i - 1]
            emit("VOLUME", i, "bullish" if up else "bearish", 2,
                 f"{sess_word}Volume spike ({tf_label})",
                 f"Volume hit ~{vol[i] / avg_v:.1f}× the recent average on a {'green' if up else 'red'} bar{thin}.", px[i])

        # sharp single-bar move
        move_thr = _MOVE_PCT_EXT if ext else _MOVE_PCT
        if ok("MOVE", i) and px[i - 1] > 0:
            chg = (px[i] - px[i - 1]) / px[i - 1] * 100
            if chg >= move_thr:
                emit("MOVE", i, "bullish", 2, f"{sess_word}Sharp jump ({tf_label})",
                     f"Popped +{chg:.1f}% in one bar to ${_round(px[i])}{thin}.", px[i])
            elif chg <= -move_thr:
                emit("MOVE", i, "bearish", 2, f"{sess_word}Sharp drop ({tf_label})",
                     f"Dropped {chg:.1f}% in one bar to ${_round(px[i])}{thin}.", px[i])

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
def _news_events(symbol: str, limit: int = 4) -> list[dict]:
    """The ticker's own recent headlines as tappable NEWS events — so a stock-
    specific catalyst (e.g. ABAT's DOE reinstatement) shows on its own tape.
    Best-effort; empty on failure."""
    try:
        from engine import alpaca_client
        news = alpaca_client.get_news(symbol, limit=limit) or []
    except Exception:
        return []
    out, seen = [], set()
    for n in news:
        head = (n.get("headline") or "").strip()
        if not head or head in seen:
            continue
        seen.add(head)
        out.append({"time": n.get("created_at") or n.get("time"),
                    "type": "NEWS", "tone": "neutral", "severity": 2,
                    "title": head, "detail": (n.get("summary") or "")[:200],
                    "url": n.get("url"), "source": n.get("source") or "news"})
    return out


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
            return _unavailable(sym, "Not enough bars in the current session yet. Check back after the open.")

        # Session bias from VWAP + the 15m EMA9/21 — ideas fire only WITH it.
        try:
            df15 = _resample(df5, "15min")
        except Exception:
            df15 = None
        bias = _session_bias(df5, df15)

        # 5m = awareness events (no ideas — too noisy). 15m = where ideas attach.
        events = _detect_tf(df5, "5m", prior_close, want_ideas=False, bias=bias)
        if df15 is not None and len(df15) >= 16:
            try:
                events += _detect_tf(df15, "15m", None, want_ideas=True, bias=bias)
            except Exception:
                pass

        events += _news_events(sym, 4)     # the ticker's own recent headlines

        # newest first, cap, and pull the most recent idea forward as the "current" idea
        events.sort(key=lambda e: e.get("time") or "", reverse=True)
        current_idea = next((e["idea"] for e in events if e.get("idea")), None)
        events = events[:_MAX_EVENTS]

        last_px = _round(df5["close"].iloc[-1])
        bull = sum(1 for e in events if e["tone"] == "bullish")
        bear = sum(1 for e in events if e["tone"] == "bearish")
        lean = "bullish" if bull > bear + 1 else "bearish" if bear > bull + 1 else "mixed"
        bias_word = {"up": "UP", "down": "DOWN"}.get(bias, "NEUTRAL")
        ct = " Counter-trend bounces are lower-odds." if bias in ("up", "down") else ""
        if events:
            summary = (f"Tape bias: {bias_word}. {len(events)} event(s) today "
                       f"({bull} bullish / {bear} bearish).{ct}")
        else:
            summary = f"Tape bias: {bias_word}. Quiet tape so far, with no notable intraday events yet."

        return {
            "available": True, "ticker": sym,
            "session_date": str(last_day),
            "as_of": (now or datetime.now(timezone.utc)).isoformat(),
            "price": last_px, "bias": bias, "lean": lean, "summary": summary,
            "current_idea": current_idea, "events": events,
            "disclaimer": _DISCLAIMER,
        }
    except Exception as e:
        logger.debug(f"[ticker_commentary] build {sym} failed: {e}")
        return _unavailable(sym, "Commentary unavailable.")
