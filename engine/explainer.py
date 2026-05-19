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


def generate(signal: dict, breakdown: dict) -> str:
    """
    Return a 2-3 sentence professional explanation of why the signal fired.
    signal keys used: ticker, direction, entry_price, stop_loss, target_one,
                      confidence_score, timeframe, strategy_type
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key and api_key != "your_anthropic_key_here":
        try:
            return _claude_explanation(api_key, signal, breakdown)
        except Exception as e:
            logger.warning(f"[explainer] Claude API error: {e} — using template")

    return _template_explanation(signal, breakdown)


# ---------------------------------------------------------------------------
# Claude explanation
# ---------------------------------------------------------------------------

def _claude_explanation(api_key: str, signal: dict, breakdown: dict) -> str:
    import anthropic

    ticker        = signal["ticker"]
    direction     = signal["direction"]
    entry         = signal.get("entry_price", signal.get("entry", "N/A"))
    sl            = signal["stop_loss"]
    t1            = signal["target_one"]
    t2            = signal["target_two"]
    score         = signal["confidence_score"]
    timeframe     = signal.get("timeframe", "1h")
    strategy_type = signal.get("strategy_type", "day_trade")

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

    confluence_text  = ", ".join(confluences) if confluences else "multiple confluences"
    strategy_focus   = _STRATEGY_FOCUS.get(strategy_type, "")
    strategy_label   = strategy_type.replace("_", " ").upper()

    prompt = (
        f"You are a professional trader. Write ONE short sentence (max 15 words) explaining "
        f"why this {strategy_label} trade is good. Use plain English — no jargon.\n"
        f"Ticker: {ticker} | Direction: {direction} | Timeframe: {timeframe}\n"
        f"Entry: {entry} | Stop: {sl} | T1: {t1} | T2: {t2} | Confidence: {score}%\n"
        f"Confluences: {confluence_text}\n"
        f"Strategy guidance: {strategy_focus}\n"
        f"Example style: 'Price bounced off key support with strong buying volume.'"
    )

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


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
