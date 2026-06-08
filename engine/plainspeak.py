"""
Server-side terminology scrubber.

Rewrites technical indicator wording to friendly language in API response TEXT
for NON-admin callers, so the raw HTTP response doesn't spell out the methodology
(MACD / EMA / VWAP / RSI / Fibonacci / golden pocket / quant / …). Admins get the
untouched technical text. Mirrors the app-side `lib/terms.ts` map.

Only transforms string leaves; never touches numbers, dict KEYS, or structure,
and leaves ticker/symbol values alone (a symbol could collide with a token).
Best-effort; never raises.

NOTE (honest scope): this hides the human-readable description, not the method —
the structured values + the chart still reveal a standard TA approach to anyone
technical. It's light obfuscation, not real IP protection.
"""
from __future__ import annotations

import re

# Multi-word / cased phrases — applied first, in order (most specific first).
_PHRASES: list[tuple[str, str]] = [
    ("MACD bullish crossover", "Momentum turned up"),
    ("MACD bearish crossover", "Momentum turned down"),
    ("MACD crossed above its signal", "Momentum turned up"),
    ("MACD crossed below its signal", "Momentum turned down"),
    ("MACD histogram", "momentum"),
    ("9/21 EMA bullish cross", "Short-term trend turned up"),
    ("9/21 EMA bearish cross", "Short-term trend turned down"),
    ("The 9 EMA crossed above the 21 EMA", "The fast average crossed above the slow"),
    ("The 9 EMA crossed below the 21 EMA", "The fast average crossed below the slow"),
    ("RSI pushed above 70", "Stretched to the upside"),
    ("RSI dropped below 30", "Washed out to the downside"),
    ("RSI overbought", "Overbought (stretched)"),
    ("RSI oversold", "Oversold (washed out)"),
    ("Reclaimed VWAP", "Back above today's avg price"),
    ("Lost VWAP", "Below today's avg price"),
    ("reclaimed VWAP", "moved back above today's average price"),
    ("lost VWAP", "fell below today's average price"),
    ("the opening range high", "the morning high"),
    ("the opening range low", "the morning low"),
    ("opening range high", "morning high"),
    ("opening range low", "morning low"),
    ("first-30m high", "morning high"),
    ("first-30m low", "morning low"),
    ("In Fibonacci pullback zone", "In key pullback area"),
    ("Fibonacci pullback zone", "key pullback area"),
    ("Fibonacci", "Key levels"),
    ("golden pocket", "key pullback area"),
    ("Golden pocket", "Key pullback area"),
    ("the 'golden pocket'", "the key pullback area"),
    ("Fib invalidation", "Level break"),
    ("TA and Quant agree", "The chart and our model agree"),
    ("TA and Quant disagree", "The chart and our model disagree"),
    ("TA / Quant disagreement", "Chart vs model disagreement"),
    ("TA & Quant", "Chart & model"),
    ("quant read", "model read"),
    ("Quant read", "Model read"),
    ("the quant", "the model"),
    ("Technicals", "The chart"),
    ("technicals", "the chart"),
    ("Multi-timeframe", "Across timeframes"),
    ("multi-timeframe", "across timeframes"),
    ("regression channel", "trend channel"),
    ("9 EMA", "fast average"),
    ("21 EMA", "slow average"),
    ("retracement", "pullback level"),
    ("1.618 extension", "upside projection"),
]

# Bare tokens — word-boundary so they don't hit substrings (e.g. "Quantity").
_TOKENS: list[tuple[str, str]] = [
    ("MACD", "momentum"),
    ("VWAP", "today's average price"),
    ("EMA", "moving average"),
    ("RSI", "momentum gauge"),
    ("ATR", "typical move"),
    ("MTF", "across timeframes"),
    ("Quant", "Model"),
    ("Fib", "Key"),
]
_TOKEN_RES = [(re.compile(r"\b" + re.escape(t) + r"\b"), p) for t, p in _TOKENS]

# Keys whose string VALUE must never be rewritten (a symbol could match a token;
# headline/summary are external verbatim text).
_SKIP_KEYS = {"ticker", "symbol", "headline", "summary"}


def plainify(text: str) -> str:
    s = text
    for t, p in _PHRASES:
        if t in s:
            s = s.replace(t, p)
    for rx, p in _TOKEN_RES:
        s = rx.sub(p, s)
    return s


# External, verbatim content — real headlines/quotes must NOT be reworded (a news
# title like "Fed signals cut" should stay intact, not become "momentum signals cut").
_VERBATIM_EVENT_TYPES = {"NEWS", "POLICY", "SOCIAL"}
_VERBATIM_EVENT_FIELDS = {"title", "detail", "summary"}


def scrub(obj):
    """Return a copy of `obj` with our technical string leaves plainified — but
    external verbatim text (news/social/policy headlines, catalyst headlines) left
    intact. Never raises."""
    try:
        if isinstance(obj, str):
            return plainify(obj)
        if isinstance(obj, list):
            return [scrub(x) for x in obj]
        if isinstance(obj, dict):
            verbatim = obj.get("type") in _VERBATIM_EVENT_TYPES
            out = {}
            for k, v in obj.items():
                if k in _SKIP_KEYS or (verbatim and k in _VERBATIM_EVENT_FIELDS):
                    out[k] = v
                else:
                    out[k] = scrub(v)
            return out
        return obj
    except Exception:
        return obj
