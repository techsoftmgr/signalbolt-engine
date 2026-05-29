"""
AI explanation generator using Anthropic Claude (claude-sonnet-4-20250514).
Falls back to a template explanation when the API key is absent or the call fails.
"""

import os
import logging

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"

_STRATEGY_FOCUS = {
    "scalping":     "Focus on momentum, tight entry/exit timing, and quick price moves.",
    "day_trade":    "Focus on intraday structure, key levels, and same-day close.",
    "swing_trade":  "Focus on higher-timeframe trend, weekly levels, and multi-day hold.",
    "options_flow": "Focus on smart money positioning via unusual options activity.",
    "dark_pool":    "Focus on institutional accumulation via large block trade detection.",
}


# The structured-narrative block keys (rendered as cards on the signal detail
# page). Mirrors the competitor layout: Market Condition + MACD + Bollinger.
NARRATIVE_KEYS = ("market_condition", "macd_note", "bb_note", "trading_strategy")


def generate(signal: dict, breakdown: dict) -> str:
    """Return just the one-sentence summary (backward-compatible)."""
    return generate_full(signal, breakdown)["summary"]


def generate_full(signal: dict, breakdown: dict) -> dict:
    """
    Return a dict with the one-line `summary` PLUS the four narrative blocks
    (market_condition, macd_note, bb_note, trading_strategy) in a single
    Claude call. Falls back to a template when the API key is absent / fails.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key and api_key != "your_anthropic_key_here":
        try:
            return _claude_full(api_key, signal, breakdown)
        except Exception as e:
            logger.warning(f"[explainer] Claude API error: {e} — using template")

    return _template_full(signal, breakdown)


def attach_narrative(signal_row: dict, breakdown: dict) -> str:
    """
    Generate the narrative once and attach it to a signal row in place:
      • signal_row["ai_explanation"]            ← summary sentence
      • signal_row["score_breakdown"]["narrative"] ← 4 blocks (no schema change)
    Returns the summary string.
    """
    full = generate_full(signal_row, breakdown)
    signal_row["ai_explanation"] = full["summary"]
    sb = signal_row.get("score_breakdown")
    if isinstance(sb, dict):
        sb["narrative"] = {k: full.get(k, "") for k in NARRATIVE_KEYS}
    return full["summary"]


# ---------------------------------------------------------------------------
# Claude explanation
# ---------------------------------------------------------------------------

def _confluence_text(breakdown: dict) -> str:
    confluences = []
    if breakdown.get("l1_smc", 0) >= 15:
        confluences.append("strong SMC structure (BOS/CHoCH with OB/FVG alignment)")
    elif breakdown.get("l1_smc", 0) >= 8:
        confluences.append("SMC structure confirmation")
    if breakdown.get("l2_technical", 0) >= 15:
        confluences.append("aligned technicals (RSI, MACD, EMA, VWAP)")
    elif breakdown.get("l2_technical", 0) >= 8:
        confluences.append("technical momentum confirmation")
    if breakdown.get("l3_sentiment", 0) >= 14:
        confluences.append("supportive news sentiment")
    if breakdown.get("l4_risk", 0) >= 7:
        confluences.append("favourable risk environment")
    return ", ".join(confluences) if confluences else "multiple confluences"


def _rr(entry, sl, t1, given) -> str:
    """Reward-to-risk as a 'N.N' string; prefer the stored value."""
    try:
        if given and float(given) > 0:
            return f"{float(given):.1f}"
    except (TypeError, ValueError):
        pass
    try:
        risk = abs(float(entry) - float(sl))
        reward = abs(float(t1) - float(entry))
        if risk > 0:
            return f"{reward / risk:.1f}"
    except (TypeError, ValueError):
        pass
    return "favourable"


def _claude_full(api_key: str, signal: dict, breakdown: dict) -> dict:
    import anthropic
    import json as _json

    ticker        = signal["ticker"]
    direction     = signal["direction"]
    entry         = signal.get("entry_price", signal.get("entry", "N/A"))
    sl            = signal.get("stop_loss")
    t1            = signal.get("target_one")
    t2            = signal.get("target_two")
    score         = signal.get("confidence_score")
    timeframe     = signal.get("timeframe", "1h")
    strategy_type = signal.get("strategy_type", "day_trade")

    confluence_text = _confluence_text(breakdown)
    strategy_focus  = _STRATEGY_FOCUS.get(strategy_type, "")
    strategy_label  = strategy_type.replace("_", " ").upper()
    rr              = _rr(entry, sl, t1, signal.get("risk_reward"))
    side            = "LONG (a buy, expecting price to rise)" if direction == "LONG" \
                      else "SHORT (a sell, expecting price to fall)"

    prompt = (
        "You are a professional trader writing a short briefing for a retail "
        "trading app. Return ONLY a JSON object (no markdown, no code fences) "
        "with exactly these string keys:\n"
        '  "summary": ONE sentence, max 15 words, why this trade is good.\n'
        '  "market_condition": 2 sentences on where price sits relative to the '
        "key support/resistance (supply/demand) zone and what reaction is "
        "expected. Mention the retest/pullback if relevant.\n"
        '  "macd_note": 1-2 sentences on what to watch on MACD to confirm this '
        "entry (cross, histogram behaviour).\n"
        '  "bb_note": 1-2 sentences on what Bollinger %B is showing and what '
        "confirms vs invalidates the setup.\n"
        '  "trading_strategy": 1-2 sentences, an actionable plan that references '
        f"the entry {entry}, stop {sl} and first target {t1} (~{rr}:1 R:R).\n"
        "Plain English; the only allowed jargon is MACD and Bollinger.\n\n"
        f"Ticker: {ticker} | Direction: {side} | Style: {strategy_label} | "
        f"Timeframe: {timeframe}\n"
        f"Entry: {entry} | Stop: {sl} | Target1: {t1} | Target2: {t2} | "
        f"Confidence: {score}%\n"
        f"Confluences: {confluence_text}\n"
        f"Strategy guidance: {strategy_focus}"
    )

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=480,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    # Be tolerant of stray code fences / prose around the JSON.
    start, end = raw.find("{"), raw.rfind("}")
    data = _json.loads(raw[start:end + 1]) if start != -1 and end != -1 else {}

    out = {k: str(data.get(k, "")).strip() for k in NARRATIVE_KEYS}
    out["summary"] = str(data.get("summary", "")).strip() or _template_explanation(signal, breakdown)
    # If the model returned no blocks at all, fall back to the template blocks.
    if not any(out[k] for k in NARRATIVE_KEYS):
        out.update(_template_blocks(signal))
    return out


# ---------------------------------------------------------------------------
# Template fallback
# ---------------------------------------------------------------------------

_TEMPLATE_LONG: dict[str, list[tuple[int, int, str]]] = {
    "scalping": [
        (15, 12, "{t} momentum surge above VWAP — quick scalp setup with tight risk."),
        (15,  0, "{t} bounced off key demand zone — fast entry with momentum."),
        ( 0, 12, "{t} EMA9/21 crossover with volume spike — scalp long signal."),
        ( 0,  0, "{t} bullish momentum detected — scalp entry at current level."),
    ],
    "day_trade": [
        (15, 12, "{t} broke above key resistance with strong momentum and buying volume."),
        (15,  0, "{t} bounced off a major support zone with institutional buying detected."),
        ( 0, 12, "{t} showing strong upside momentum across multiple timeframes."),
        ( 0,  0, "{t} set up a bullish entry with price at a key demand level."),
    ],
    "swing_trade": [
        (15, 12, "{t} higher-timeframe trend resumption — multi-day long opportunity."),
        (15,  0, "{t} weekly demand zone holding — swing long with wide targets."),
        ( 0, 12, "{t} trend alignment across 4H and daily — swing continuation."),
        ( 0,  0, "{t} bullish swing setup at key higher-timeframe level."),
    ],
    "options_flow": [
        ( 0,  0, "{t} unusual call activity detected — smart money positioning long."),
    ],
    "dark_pool": [
        ( 0,  0, "{t} large block trade at key level — institutional accumulation detected."),
    ],
}

_TEMPLATE_SHORT: dict[str, list[tuple[int, int, str]]] = {
    "scalping": [
        (15, 12, "{t} rejection at VWAP resistance — fast short scalp setup."),
        (15,  0, "{t} supply zone rejection — quick short with tight stop."),
        ( 0, 12, "{t} EMA bearish crossover with volume — scalp short signal."),
        ( 0,  0, "{t} bearish momentum detected — scalp short at current level."),
    ],
    "day_trade": [
        (15, 12, "{t} broke below key support with strong selling pressure and momentum."),
        (15,  0, "{t} rejected from a major resistance zone with institutional selling detected."),
        ( 0, 12, "{t} showing strong downside momentum across multiple timeframes."),
        ( 0,  0, "{t} set up a bearish entry with price at a key supply level."),
    ],
    "swing_trade": [
        (15, 12, "{t} higher-timeframe downtrend resumption — multi-day short opportunity."),
        (15,  0, "{t} weekly supply zone holding — swing short with wide targets."),
        ( 0, 12, "{t} bearish trend alignment across 4H and daily — swing continuation."),
        ( 0,  0, "{t} bearish swing setup at key higher-timeframe supply level."),
    ],
    "options_flow": [
        ( 0,  0, "{t} unusual put activity detected — smart money positioning short."),
    ],
    "dark_pool": [
        ( 0,  0, "{t} large block trade at resistance — institutional distribution detected."),
    ],
}


def _template_explanation(signal: dict, breakdown: dict) -> str:
    ticker        = signal["ticker"]
    direction     = signal["direction"]
    strategy_type = signal.get("strategy_type", "day_trade")
    l1 = breakdown.get("l1_smc", 0)
    l2 = breakdown.get("l2_technical", 0)

    templates = _TEMPLATE_LONG if direction == "LONG" else _TEMPLATE_SHORT
    options   = templates.get(strategy_type, templates.get("day_trade", []))

    for min_l1, min_l2, tpl in options:
        if l1 >= min_l1 and l2 >= min_l2:
            return tpl.format(t=ticker)

    # Final fallback
    side = "bullish" if direction == "LONG" else "bearish"
    return f"{ticker} {side} setup detected — {strategy_type.replace('_', ' ')} signal."


# ---------------------------------------------------------------------------
# Template narrative blocks (no-API fallback)
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    try:
        return f"${float(v):.2f}"
    except (TypeError, ValueError):
        return "the level"


def _template_blocks(signal: dict) -> dict:
    """Direction-aware narrative blocks used when the Claude API is unavailable."""
    ticker = signal.get("ticker", "This name")
    entry  = _fmt(signal.get("entry_price", signal.get("entry")))
    sl     = _fmt(signal.get("stop_loss"))
    t1     = _fmt(signal.get("target_one"))
    rr     = _rr(signal.get("entry_price", signal.get("entry")),
                 signal.get("stop_loss"), signal.get("target_one"),
                 signal.get("risk_reward"))

    if signal.get("direction") == "LONG":
        return {
            "market_condition": (
                f"{ticker} is pulling back into a key demand zone after an advance. "
                "Price is retesting this level — holding here keeps the bullish "
                "structure intact and sets up a long reaction."
            ),
            "macd_note": (
                "Watch the MACD histogram stop falling and tick higher, or the MACD "
                "line cross up through its signal — early confirmation buyers are "
                "stepping back in at the zone."
            ),
            "bb_note": (
                "Bollinger %B near or below 0 means price has stretched to the lower "
                "band. A turn back up supports the bounce; failure to recover warns "
                "the zone is giving way."
            ),
            "trading_strategy": (
                f"Enter long near {entry} with a stop below {sl} (zone invalidation) "
                f"and a first target at {t1} — about {rr}:1 reward-to-risk. Scale out "
                "into strength or trail the stop up."
            ),
        }
    return {
        "market_condition": (
            f"{ticker} is rallying back up into a supply zone — a former support that "
            "broke and now acts as resistance. This retest offers a short if sellers "
            "defend the level."
        ),
        "macd_note": (
            "Watch the MACD histogram fade and roll over, or the MACD line cross down "
            "through its signal — confirmation the bounce is losing steam into "
            "resistance."
        ),
        "bb_note": (
            "Bollinger %B pushing toward the upper band (0.5–1.0) shows price "
            "stretching up. A rejection back down confirms sellers; a clean break "
            "above warns the short is failing."
        ),
        "trading_strategy": (
            f"Enter short near {entry} with a stop above {sl} (zone invalidation) and "
            f"a first target at {t1} — about {rr}:1 reward-to-risk. Cover into weakness "
            "or trail the stop down."
        ),
    }


def _template_full(signal: dict, breakdown: dict) -> dict:
    return {"summary": _template_explanation(signal, breakdown), **_template_blocks(signal)}
