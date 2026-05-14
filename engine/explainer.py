"""
AI explanation generator using Anthropic Claude (claude-sonnet-4-20250514).
Falls back to a template explanation when the API key is absent or the call fails.
"""

import os
import logging

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"


def generate(signal: dict, breakdown: dict) -> str:
    """
    Return a 2-3 sentence professional explanation of why the signal fired.
    signal keys used: ticker, direction, entry_price, stop_loss, target_one,
                      confidence_score, timeframe
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

    ticker = signal["ticker"]
    direction = signal["direction"]
    entry = signal.get("entry_price", signal.get("entry", "N/A"))
    sl = signal["stop_loss"]
    t1 = signal["target_one"]
    score = signal["confidence_score"]
    timeframe = signal.get("timeframe", "1h")

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

    confluence_text = ", ".join(confluences) if confluences else "multiple SMC confluences"

    prompt = (
        f"Write ONE short sentence (max 15 words) explaining why this trade is good. "
        f"Use plain English — no jargon, no hype. Just the key reason.\n"
        f"Ticker: {ticker} | Direction: {direction} | Confluences: {confluence_text}\n"
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

def _template_explanation(signal: dict, breakdown: dict) -> str:
    ticker   = signal["ticker"]
    direction = signal["direction"]
    l1 = breakdown.get("l1_smc", 0)
    l2 = breakdown.get("l2_technical", 0)

    if direction == "LONG":
        if l1 >= 15 and l2 >= 12:
            return f"{ticker} broke above key resistance with strong momentum and buying volume."
        elif l1 >= 15:
            return f"{ticker} bounced off a major support zone with institutional buying detected."
        elif l2 >= 12:
            return f"{ticker} showing strong upside momentum across multiple timeframes."
        else:
            return f"{ticker} set up a bullish entry with price at a key demand level."
    else:
        if l1 >= 15 and l2 >= 12:
            return f"{ticker} broke below key support with strong selling pressure and momentum."
        elif l1 >= 15:
            return f"{ticker} rejected from a major resistance zone with institutional selling detected."
        elif l2 >= 12:
            return f"{ticker} showing strong downside momentum across multiple timeframes."
        else:
            return f"{ticker} set up a bearish entry with price at a key supply level."
