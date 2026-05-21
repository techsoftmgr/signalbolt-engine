"""
Expo Push Notification dispatcher.
Fetches all push tokens from Supabase profiles and sends via Expo Push API.

Token list is cached for 5 minutes so rapid-fire notifications (e.g. T1 hit
+ reversal on 5 active signals) do not hammer the DB with repeated queries.
"""

import logging
import os
import time

import requests
from supabase import create_client, Client

logger = logging.getLogger(__name__)

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"

# ── Token cache ───────────────────────────────────────────────
# A full profiles scan per notification is expensive at scale.
# Cache for 5 minutes — new tokens are picked up within 1 cycle.
_token_cache: list[str] = []
_token_cache_ts: float  = 0.0
_TOKEN_CACHE_TTL        = 300   # 5 minutes

# Single Supabase client reused for the lifetime of the process
_sb_client: Client | None = None


def _supabase() -> Client:
    global _sb_client
    if _sb_client is None:
        key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
        _sb_client = create_client(os.environ["SUPABASE_URL"], key)
    return _sb_client


def _get_tokens() -> list[str]:
    global _token_cache, _token_cache_ts
    now = time.monotonic()
    if now - _token_cache_ts < _TOKEN_CACHE_TTL:
        return _token_cache        # serve from cache

    try:
        rows = (
            _supabase()
            .table("profiles")
            .select("push_token")
            .neq("push_token", None)
            .execute()
            .data
        )
        tokens = [r["push_token"] for r in rows if r.get("push_token")]
        _token_cache    = tokens
        _token_cache_ts = now
        return tokens
    except Exception as e:
        logger.error(f"[push] Failed to fetch push tokens: {e}")
        return _token_cache   # return stale cache rather than empty on transient error


def send_signal_alert(
    ticker: str,
    direction: str,
    confidence: int,
    signal_type: str = "stock",
    signal_id: str | None = None,
) -> None:
    """Send a push notification to all registered devices when a signal fires."""
    tokens = _get_tokens()
    if not tokens:
        logger.info("[push] No push tokens registered — skipping notification")
        return

    emoji = "📈" if direction == "LONG" else "📉"
    type_label = "Options" if signal_type == "option" else "Stock"

    # Include signal_id so tapping the notification deep-links straight to the signal card
    notif_data: dict = {"ticker": ticker, "direction": direction, "type": signal_type}
    if signal_id:
        notif_data["signal_id"] = signal_id

    messages = [
        {
            "to":    token,
            "title": f"{emoji} New {type_label} Signal: {ticker}",
            "body":  f"{direction} · {confidence}% confidence · Tap to view details",
            "data":  notif_data,
            "sound": "default",
            "badge": 1,
        }
        for token in tokens
        if token.startswith("ExponentPushToken[")
    ]

    if not messages:
        logger.info("[push] No valid Expo tokens found")
        return

    try:
        resp = requests.post(
            EXPO_PUSH_URL,
            json=messages,
            headers={
                "Accept":       "application/json",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        data = resp.json()
        errors = [
            r.get("details", {}).get("error")
            for r in data.get("data", [])
            if r.get("status") == "error"
        ]
        if errors:
            logger.warning(f"[push] Some notifications failed: {errors}")
        else:
            logger.info(f"[push] Sent {len(messages)} notification(s) for {ticker} {direction}")
    except Exception as e:
        logger.error(f"[push] Failed to send notifications: {e}")


def _send_raw(title: str, body: str, data: dict | None = None) -> None:
    """
    Send a push notification with a custom title/body to all registered devices.
    Used by signal_monitor.py for close events, reversals, and breakeven moves.
    """
    tokens = _get_tokens()
    if not tokens:
        logger.info("[push] No push tokens registered — skipping raw notification")
        return

    messages = [
        {
            "to":    token,
            "title": title,
            "body":  body,
            "data":  data or {},
            "sound": "default",
            "badge": 1,
        }
        for token in tokens
        if token.startswith("ExponentPushToken[")
    ]

    if not messages:
        logger.info("[push] No valid Expo tokens for raw notification")
        return

    try:
        resp = requests.post(
            EXPO_PUSH_URL,
            json=messages,
            headers={
                "Accept":       "application/json",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        result = resp.json()
        errors = [
            r.get("details", {}).get("error")
            for r in result.get("data", [])
            if r.get("status") == "error"
        ]
        if errors:
            logger.warning(f"[push] Raw notification errors: {errors}")
        else:
            logger.info(f"[push] Raw notification sent to {len(messages)} device(s): {title}")
    except Exception as e:
        logger.error(f"[push] Failed to send raw notification: {e}")
