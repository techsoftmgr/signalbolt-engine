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
    "community_buzz": True,   # watchlist-scoped social-buzz spike alerts
    "cycle_signals":  True,   # turnaround Buy-Zone / Peak distribution alerts
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


def _tokens_for(pref_key: str, default: bool = True) -> list[str]:
    """
    Return only tokens where the user has enabled the given notification type.

    `default` controls behavior when the user hasn't set this pref yet:
      - True  → opt-out (most prefs work this way)
      - False → opt-in  (used for noisy alerts like block_prints)
    """
    return [
        p["token"]
        for p in _get_profiles()
        if p["prefs"].get(pref_key, default)
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


def send_stop_protected_alert(
    ticker: str,
    direction: str,
    new_stop: float,
    locked_pct: float,
    signal_id: str | None = None,
) -> None:
    """Push ONCE when a signal's stop is first trailed to breakeven-or-better.

    The monitors raise the stop silently as price runs; a user who isn't watching
    never learns their downside is now protected (and that their broker stop is
    stale). Gated by the t1_breakeven pref ("stop moved to breakeven"). Sent only
    on the single crossing (handled by the caller), so no spam on later ratchets.
    """
    try:
        tokens = _tokens_for("t1_breakeven")
        if not tokens:
            return
        notif_data: dict = {"type": "t1_breakeven", "ticker": ticker, "direction": direction}
        if signal_id:
            notif_data["signal_id"] = signal_id
        messages = [
            {
                "to":    token,
                "title": f"🔒 {ticker} stop raised — now risk-free",
                "body":  (f"We moved your stop to ${new_stop:.2f}, locking +{locked_pct:.1f}%. "
                          f"Downside is protected — update your broker stop to match."),
                "data":  notif_data,
                "sound": "default",
                "badge": 1,
            }
            for token in tokens
        ]
        _dispatch(messages, f"{ticker} stop→BE")
    except Exception as e:
        logger.debug(f"[push] stop_protected alert failed for {ticker}: {e}")


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
    "block_print":    "block_prints",  # new — opt-in whale alerts
}


def send_block_print_alert(ticker: str, size: int, price: float, direction: str = "-") -> None:
    """
    Whale-watch alert: institutional block trade just printed on `ticker`.
    `direction` is tick-rule classification: 'B' buy-initiated, 'S' sell-
    initiated, '-' neutral / unknown.

    Opt-in (default off) — uses _tokens_for(..., default=False) directly so
    users have to flip the pref ON to receive these. Fire-and-forget.
    """
    try:
        tokens = _tokens_for("block_prints", default=False)
        if not tokens:
            return
        notional_m = (size * price) / 1_000_000
        if direction == "B":
            label, emoji = "BUY", "🟢"
        elif direction == "S":
            label, emoji = "SELL", "🔴"
        else:
            label, emoji = "block", "🐋"
        messages = [
            {
                "to":    t,
                "title": f"{emoji} {ticker} {label} block",
                "body":  f"{size:,} shares @ ${price:.2f}  ·  ~${notional_m:.1f}M",
                "data":  {"type": "block_print", "ticker": ticker, "size": size,
                          "price": price, "direction": direction},
                "sound": "default",
                "badge": 1,
            }
            for t in tokens
        ]
        _dispatch(messages, f"BLOCK {direction} {ticker}")
    except Exception as e:
        logger.debug(f"[push] block_print alert failed for {ticker}: {e}")


def send_buzz_spike_alert(
    ticker: str,
    change_pct: float | None = None,
    mentions: int | None = None,
    sb: Client | None = None,
) -> int:
    """
    Notify users who WATCH `ticker` that its social buzz is spiking.

    Watchlist-scoped (NOT a broadcast) so it stays relevant instead of spammy —
    only users with `ticker` on their watchlist and the `community_buzz` pref on
    (default on) get pinged. Returns the number of notifications dispatched.
    Fire-and-forget; the caller handles per-day dedup.
    """
    try:
        client = sb or _supabase()
        watchers = (
            client.table("watchlist").select("user_id").eq("ticker", ticker).execute().data
        ) or []
        user_ids = list({w["user_id"] for w in watchers if w.get("user_id")})
        if not user_ids:
            return 0

        prof = (
            client.table("profiles")
            .select("push_token, notification_prefs")
            .in_("id", user_ids)
            .neq("push_token", None)
            .execute()
            .data
        ) or []
        tokens = [
            p["push_token"]
            for p in prof
            if p.get("push_token", "").startswith("ExponentPushToken[")
            and {**_DEFAULT_PREFS, **(p.get("notification_prefs") or {})}.get("community_buzz", True)
        ]
        if not tokens:
            return 0

        chg = f" (+{change_pct:.0f}% mentions)" if change_pct is not None else ""
        messages = [
            {
                "to":    t,
                "title": f"🔥 {ticker} buzz spiking",
                "body":  f"{ticker} is heating up on social{chg}. Tap to see why.",
                "data":  {"type": "community_buzz", "ticker": ticker},
                "sound": "default",
                "badge": 1,
            }
            for t in tokens
        ]
        _dispatch(messages, f"BUZZ {ticker}")
        return len(messages)
    except Exception as e:
        logger.debug(f"[push] buzz spike alert failed for {ticker}: {e}")
        return 0


def send_cycle_alert(ticker: str, kind: str, sb: Client | None = None) -> int:
    """
    Notify users who WATCH `ticker` of a confirmed cycle signal:
      kind="turnaround" → swing-low Buy Zone (reversal confirmed)
      kind="peak"       → swing-high / distribution top (take profit / hedge)

    Watchlist-scoped (same targeting as the buzz alert), respects the
    'cycle_signals' pref (default on). Returns the number dispatched. The caller
    handles per-day dedup.
    """
    try:
        client = sb or _supabase()
        watchers = (
            client.table("watchlist").select("user_id").eq("ticker", ticker).execute().data
        ) or []
        user_ids = list({w["user_id"] for w in watchers if w.get("user_id")})
        if not user_ids:
            return 0
        prof = (
            client.table("profiles")
            .select("push_token, notification_prefs")
            .in_("id", user_ids)
            .neq("push_token", None)
            .execute()
            .data
        ) or []
        tokens = [
            p["push_token"]
            for p in prof
            if p.get("push_token", "").startswith("ExponentPushToken[")
            and {**_DEFAULT_PREFS, **(p.get("notification_prefs") or {})}.get("cycle_signals", True)
        ]
        if not tokens:
            return 0
        if kind == "turnaround":
            title = f"🔄 {ticker} — Turnaround Buy Zone"
            body  = f"{ticker} confirmed a reversal off the lows. Tap for the setup."
        else:
            title = f"🔻 {ticker} — Peak / Distribution"
            body  = f"{ticker} looks topped — consider taking profit / hedging. Tap for details."
        messages = [
            {
                "to":    t,
                "title": title,
                "body":  body,
                "data":  {"type": "cycle_signal", "ticker": ticker, "kind": kind},
                "sound": "default",
                "badge": 1,
            }
            for t in tokens
        ]
        _dispatch(messages, f"CYCLE {kind} {ticker}")
        return len(messages)
    except Exception as e:
        logger.debug(f"[push] cycle alert failed for {ticker}: {e}")
        return 0


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
