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

# ── Token + prefs cache ───────────────────────────────────────
# A full profiles scan per notification is expensive at scale.
# Cache for 5 minutes — new tokens/prefs are picked up within 1 cycle.
#
# Each entry: { "token": "ExponentPushToken[...]", "prefs": { "new_signals": True, ... } }
_profile_cache: list[dict] = []
_profile_cache_ts: float   = 0.0
_TOKEN_CACHE_TTL           = 300   # 5 minutes

_DEFAULT_PREFS = {
    "new_signals":    True,
    "target_hit":     True,
    "stop_hit":       True,
    "t1_breakeven":   True,
    "market_open":    False,
    "weekly_summary": True,
}

# Single Supabase client reused for the lifetime of the process
_sb_client: Client | None = None


def _supabase() -> Client:
    global _sb_client
    if _sb_client is None:
        key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
        _sb_client = create_client(os.environ["SUPABASE_URL"], key)
    return _sb_client


def _get_profiles() -> list[dict]:
    """
    Return cached list of {token, prefs} dicts. Refresh every 5 minutes.

    Retries up to 3 times (backoff 1s/2s/4s) before logging a single error
    and falling back to the stale cache. This prevents transient Supabase
    blips from spamming the logs and from clearing the working cache.
    """
    global _profile_cache, _profile_cache_ts
    now = time.monotonic()
    if now - _profile_cache_ts < _TOKEN_CACHE_TTL:
        return _profile_cache   # serve from cache

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            rows = (
                _supabase()
                .table("profiles")
                .select("push_token, notification_prefs")
                .neq("push_token", None)
                .execute()
                .data
            )
            profiles = [
                {
                    "token": r["push_token"],
                    "prefs": {**_DEFAULT_PREFS, **(r.get("notification_prefs") or {})},
                }
                for r in rows
                if r.get("push_token") and r["push_token"].startswith("ExponentPushToken[")
            ]
            _profile_cache    = profiles
            _profile_cache_ts = now
            return profiles
        except Exception as e:
            last_error = e
            if attempt < 2:
                logger.warning(
                    "[push] token fetch failed attempt=%s; retrying in %ss",
                    attempt + 1, 2 ** attempt,
                )
                time.sleep(2 ** attempt)

    logger.error(f"[push] Failed to fetch push profiles after retries: {last_error}")
    return _profile_cache   # return stale cache on persistent error


def _tokens_for(pref_key: str) -> list[str]:
    """Return only tokens where the user has enabled the given notification type."""
    return [
        p["token"]
        for p in _get_profiles()
        if p["prefs"].get(pref_key, True)   # default True if pref missing
    ]


def send_signal_alert(
    ticker: str,
    direction: str,
    confidence: int,
    signal_type: str = "stock",
    signal_id: str | None = None,
) -> None:
    """Send a new-signal push to users who have new_signals pref enabled."""
    tokens = _tokens_for("new_signals")
    if not tokens:
        logger.info("[push] No tokens with new_signals enabled — skipping")
        return

    emoji = "📈" if direction == "LONG" else "📉"
    type_label = "Options" if signal_type == "option" else "Stock"

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
    ]

    _dispatch(messages, f"{ticker} {direction}")


# Maps notification data["type"] → notification_prefs key
# Types not listed here are sent to all users (no pref filter)
_TYPE_TO_PREF: dict[str, str] = {
    "signal_closed":  "target_hit",    # win case — also covers target_hit pref
    "t1_breakeven":   "t1_breakeven",
    "scalp_expired":  "stop_hit",
    "market_close":   "stop_hit",
    "eod_warning":    "target_hit",
    "book_profit":    "target_hit",
    "reversal":       "stop_hit",
}


def _send_raw(
    title: str,
    body: str,
    data: dict | None = None,
    pref_key: str | None = None,
) -> None:
    """
    Send a push notification with a custom title/body.
    Automatically respects user notification preferences:
      • If pref_key is provided explicitly, use it.
      • Otherwise infer from data["type"] via _TYPE_TO_PREF.
      • If no mapping, send to all registered tokens.
    Also handles stop_hit vs target_hit split for signal_closed events.
    """
    payload = data or {}
    notif_type = payload.get("type", "")

    # Special case: signal_closed result=loss → stop_hit pref
    if notif_type == "signal_closed" and payload.get("result") == "loss":
        resolved_pref = "stop_hit"
    else:
        resolved_pref = pref_key or _TYPE_TO_PREF.get(notif_type)

    tokens = _tokens_for(resolved_pref) if resolved_pref else [p["token"] for p in _get_profiles()]
    if not tokens:
        logger.info(f"[push] No eligible tokens for type='{notif_type}' pref='{resolved_pref}' — skipping")
        return

    messages = [
        {
            "to":    token,
            "title": title,
            "body":  body,
            "data":  payload,
            "sound": "default",
            "badge": 1,
        }
        for token in tokens
    ]

    _dispatch(messages, title)


def _dispatch(messages: list[dict], label: str) -> None:
    """Fire messages to Expo Push API and log results."""
    if not messages:
        logger.info(f"[push] No messages to dispatch for: {label}")
        return
    try:
        resp = requests.post(
            EXPO_PUSH_URL,
            json=messages,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=10,
        )
        result = resp.json()
        errors = [
            r.get("details", {}).get("error")
            for r in result.get("data", [])
            if r.get("status") == "error"
        ]
        if errors:
            logger.warning(f"[push] Errors for '{label}': {errors}")
        else:
            logger.info(f"[push] Sent {len(messages)} notification(s): {label}")
    except Exception as e:
        logger.error(f"[push] Dispatch failed for '{label}': {e}")
