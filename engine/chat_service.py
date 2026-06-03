"""
Stock-chat assistant (cost-bounded, grounded).
================================================
Answers user questions about a ticker by EXPLAINING the app's own analysis —
the quant verdict, the Expert Read, and any active signal's levels — rather than
doing open-ended (expensive, hallucination-prone) analysis.

Cost controls:
  - Haiku model (cheap/fast)
  - Prompt caching on the static instructions + grounded context block
  - Compact context (a few hundred tokens, not raw JSON/bars)
  - History truncated to the last few turns
  - max_tokens cap on the answer
Per-user daily quota is enforced by the caller (main.py), not here.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("signalbolt.chat")

MODEL          = "claude-haiku-4-5-20251001"   # cheap; matches explainer.py
_MAX_TOKENS    = 420
_MAX_HISTORY   = 6      # last N turns re-sent each request
_MAX_MSG_CHARS = 600    # clamp a single user message

_DISCLAIMER = "Educational only — not financial advice."

_INSTRUCTIONS = (
    "You are SignalBolt's stock assistant. You help a retail trader understand a "
    "specific ticker by EXPLAINING SignalBolt's own analysis for it: the quant "
    "verdict, the Expert Read (technical bias + trade idea), and any ACTIVE signal's "
    "entry/stop/targets. Rules:\n"
    "- Ground every answer in the CONTEXT below. If the context lacks something, say "
    "so plainly — do NOT invent price targets, levels, or signals.\n"
    "- When asked 'what entry/should I buy', describe the engine's active signal "
    "levels (entry/stop/targets) and the Expert Read idea; explain the reasoning. Do "
    "NOT tell the user to buy/sell — present what the engine sees and the risk.\n"
    "- Be concise (a few sentences). Plain language. No tables unless asked.\n"
    "- You are not a licensed advisor. Always frame as educational. If asked for a "
    "guarantee or 'will it go up', explain nobody can know and point to the risk levels.\n"
    f"- End with a brief reminder: '{_DISCLAIMER}'"
)


def _build_context(sb, ticker: str) -> str:
    """Compact, grounded context block from the app's existing analysis."""
    tk = (ticker or "").upper()
    lines = [f"TICKER: {tk}"]

    # Quant verdict (cached scan)
    try:
        from engine import quant_score_service as _qs
        row, _asof = _qs.cached_score(tk)
        if row:
            lines.append(
                f"QUANT: trendScore={row.get('trendScore')} setup={row.get('setupType')} "
                f"ma20={row.get('ma20')} atr%={row.get('atrPct')} dayChg%={row.get('dayChangePct')} "
                f"peakStage={row.get('peakStage')} turnStage={row.get('turnaroundStage')}"
            )
    except Exception as e:
        logger.debug(f"[chat] quant context failed {tk}: {e}")

    # Expert Read (chart_read)
    try:
        from engine import chart_read as _cr
        cr = _cr.analyze(tk)
        if cr:
            lines.append(
                f"EXPERT_READ: bias={cr.get('taBias')} short_term={cr.get('short_term')} "
                f"agreement={cr.get('agreement')}"
            )
            idea = cr.get("idea")
            if idea:
                lines.append(f"TRADE_IDEA: {str(idea)[:280]}")
            narr = cr.get("narrative")
            if narr:
                lines.append(f"NARRATIVE: {str(narr)[:320]}")
            pats = cr.get("patterns")
            if pats:
                lines.append(f"PATTERNS: {str(pats)[:160]}")
    except Exception as e:
        logger.debug(f"[chat] expert-read context failed {tk}: {e}")

    # Active signals on this ticker
    try:
        sigs = (
            sb.table("signals")
            .select("direction,entry_price,stop_loss,target_one,target_two,strategy_type,confidence_score")
            .eq("ticker", tk).eq("status", "active").limit(5).execute()
        ).data or []
        for s in sigs:
            lines.append(
                f"ACTIVE_SIGNAL: {s.get('direction')} {s.get('strategy_type')} "
                f"entry={s.get('entry_price')} stop={s.get('stop_loss')} "
                f"t1={s.get('target_one')} t2={s.get('target_two')} conf={s.get('confidence_score')}"
            )
        if not sigs:
            lines.append("ACTIVE_SIGNAL: none right now")
    except Exception as e:
        logger.debug(f"[chat] active-signal context failed {tk}: {e}")

    return "\n".join(lines)


def _clean_history(history) -> list[dict]:
    """Keep the last _MAX_HISTORY valid {role, content} turns."""
    out = []
    for m in (history or []):
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content[:1200]})
    return out[-_MAX_HISTORY:]


def answer(sb, ticker: str, user_message: str, history=None) -> dict:
    """Return {ok, answer, usage}. Grounded, Haiku, prompt-cached."""
    msg = (user_message or "").strip()[:_MAX_MSG_CHARS]
    if not msg:
        return {"ok": False, "answer": "Ask me something about this stock.", "usage": {}}

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_anthropic_key_here":
        return {"ok": False, "answer": "Chat is temporarily unavailable.", "usage": {}}

    context = _build_context(sb, ticker)
    messages = _clean_history(history) + [{"role": "user", "content": msg}]

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=MODEL,
            max_tokens=_MAX_TOKENS,
            system=[
                {"type": "text", "text": _INSTRUCTIONS},
                # Cache the grounded context so multi-turn chat re-uses it cheaply.
                {"type": "text", "text": f"CONTEXT for {ticker}:\n{context}",
                 "cache_control": {"type": "ephemeral"}},
            ],
            messages=messages,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        u = resp.usage
        usage = {
            "input_tokens":  getattr(u, "input_tokens", 0),
            "output_tokens": getattr(u, "output_tokens", 0),
            "cache_read":    getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write":   getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        logger.info(f"[chat] {ticker} answered — usage={usage}")
        return {"ok": True, "answer": text or "I couldn't form an answer — try rephrasing.", "usage": usage}
    except Exception as e:
        logger.warning(f"[chat] {ticker} answer failed: {e}")
        return {"ok": False, "answer": "Chat hit an error — please try again.", "usage": {}}
