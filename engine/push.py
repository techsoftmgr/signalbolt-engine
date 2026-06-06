"""
Expo Push Notification dispatcher.
Fetches all push tokens from Supabase profiles and sends via Expo Push API.

Token list is cached for 5 minutes so rapid-fire notifications (e.g. T1 hit
+ reversal on 5 active signals) do not hammer the DB with repeated queries.
"""

import logging
import os
import time
from datetime import datetime, timezone

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
    "watchlist_alerts": True, # watched ticker changed state (buy zone / topping / breakout / trend lost)
    "breakdown_alerts": True, # universe-wide heavy-selling / breakdown-risk alerts
    "breakout_alerts":  True, # universe-wide breakout alerts (broke 20-day high on vol)
    "accumulation_alerts": True, # universe-wide unusual-buying (heavy up-volume) alerts
    "premarket_gap_alerts": True, # premarket disaster-gap heads-up for open overnight positions (notification only)
    "commentary_alerts": True,    # live intraday events on a watched ticker (MACD cross, breakout, VWAP, gap, sharp move)
}

# Single Supabase client reused for the lifetime of the process
_sb_client: Client | None = None


def _supabase() -> Client:
    global _sb_client
    if _sb_client is None:
        key = os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SECRET_KEY"]
        _sb_client = create_client(os.environ["SUPABASE_URL"], key)
    return _sb_client


def _valid_token(tok) -> bool:
    return isinstance(tok, str) and tok.startswith("ExponentPushToken[")


def _merge_devices(ptoks: list[dict], profiles: list[dict]) -> list[dict]:
    """PURE — merge the push_tokens table (multi-device) with the legacy
    profiles.push_token into one deduped device list:
        [{user_id, token, prefs, email}]
    push_tokens wins; a legacy profiles.push_token is included only if that token
    isn't already present. Non-Expo tokens are dropped. Deterministic + testable.
    """
    pmap = {p.get("id"): p for p in (profiles or [])}
    out: list[dict] = []
    seen: set = set()

    def _add(user_id, token, prof):
        if not _valid_token(token) or token in seen:
            return
        out.append({
            "user_id": user_id,
            "token":   token,
            "prefs":   {**_DEFAULT_PREFS, **((prof or {}).get("notification_prefs") or {})},
            "email":   (prof or {}).get("email"),
        })
        seen.add(token)

    for pt in (ptoks or []):
        _add(pt.get("user_id"), pt.get("token"), pmap.get(pt.get("user_id")))
    for p in (profiles or []):          # legacy fallback
        _add(p.get("id"), p.get("push_token"), p)
    return out


def _device_rows() -> list[dict]:
    """
    Cached list of {user_id, token, prefs, email} for ALL registered devices —
    the UNION of the push_tokens table (one user → many devices) and the legacy
    profiles.push_token. Refresh every 5 min; retries with backoff; serves stale
    cache on persistent failure (a Supabase blip must not silence push).
    """
    global _profile_cache, _profile_cache_ts
    now = time.monotonic()
    if now - _profile_cache_ts < _TOKEN_CACHE_TTL:
        return _profile_cache

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            profiles = (
                _supabase().table("profiles")
                .select("id, push_token, notification_prefs, email")
                .execute().data
            ) or []
            try:
                ptoks = (
                    _supabase().table("push_tokens").select("user_id, token").execute().data
                ) or []
            except Exception:
                ptoks = []   # table not migrated yet → legacy profiles-only
            devices = _merge_devices(ptoks, profiles)
            _profile_cache    = devices
            _profile_cache_ts = now
            return devices
        except Exception as e:
            last_error = e
            if attempt < 2:
                logger.warning(
                    "[push] device fetch failed attempt=%s; retrying in %ss",
                    attempt + 1, 2 ** attempt,
                )
                time.sleep(2 ** attempt)

    logger.error(f"[push] Failed to fetch devices after retries: {last_error}")
    return _profile_cache   # stale cache on persistent error


def _get_profiles() -> list[dict]:
    """Back-compat shim: [{token, prefs}] across ALL devices."""
    return [{"token": d["token"], "prefs": d["prefs"]} for d in _device_rows()]


def _tokens_for(pref_key: str, default: bool = True) -> list[str]:
    """
    Tokens (all devices) where the user has enabled the given notification type.
    `default` controls users who haven't set this pref: True → opt-out, False →
    opt-in (noisy alerts like block_prints).
    """
    return [d["token"] for d in _device_rows() if d["prefs"].get(pref_key, default)]


def _tokens_for_users(user_ids, pref_key: str | None = None, default: bool = True) -> list[str]:
    """Tokens for a set of user_ids (ALL of each user's devices), optionally
    pref-filtered. Used by watchlist / buzz / cycle (audience-scoped) alerts."""
    uid = {u for u in (user_ids or []) if u}
    if not uid:
        return []
    out = []
    for d in _device_rows():
        if d["user_id"] not in uid:
            continue
        if pref_key is not None and not d["prefs"].get(pref_key, default):
            continue
        out.append(d["token"])
    return out


def register_device(user_id: str, token: str, platform: str | None = None) -> bool:
    """
    Service-role upsert of a device push token (multi-device aware). A given Expo
    token belongs to exactly ONE install, so we REASSIGN it to this user if it was
    previously registered to another account (re-login on the same device). Also
    mirrors to profiles.push_token for back-compat. Returns True on success.
    """
    if not _valid_token(token):
        return False
    try:
        sb = _supabase()
        sb.table("push_tokens").delete().eq("token", token).execute()   # reassign
        sb.table("push_tokens").insert({
            "user_id":   user_id,
            "token":     token,
            "platform":  platform,
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }).execute()
        try:
            sb.table("profiles").update({"push_token": token}).eq("id", user_id).execute()
        except Exception:
            pass   # back-compat mirror is best-effort
        global _profile_cache_ts
        _profile_cache_ts = 0.0   # invalidate cache so the new token is used now
        return True
    except Exception as e:
        logger.error(f"[push] register_device failed for {user_id}: {e}")
        return False


def _prune_dead_tokens(tokens: list[str]) -> None:
    """Delete tokens Expo reported as DeviceNotRegistered (app uninstalled / token
    rotated) from push_tokens + clear any legacy profiles.push_token match."""
    try:
        sb = _supabase()
        for t in tokens:
            if not t:
                continue
            try: sb.table("push_tokens").delete().eq("token", t).execute()
            except Exception: pass
            try: sb.table("profiles").update({"push_token": None}).eq("push_token", t).execute()
            except Exception: pass
        global _profile_cache_ts
        _profile_cache_ts = 0.0
        logger.info(f"[push] pruned {len(tokens)} dead token(s)")
    except Exception as e:
        logger.debug(f"[push] prune dead tokens failed: {e}")


def _record_alert(
    type_: str,
    ticker: str | None,
    title: str,
    body: str,
    stage: str | None = None,
    data: dict | None = None,
    sb: Client | None = None,
) -> None:
    """Persist an alert to the shared in-app Alerts feed (the `alerts` table).

    Best-effort and INDEPENDENT of push delivery — the in-app Alerts tab must
    populate even when no device has a push token (e.g. FCM not configured on a
    standalone Android build). Never raise; telemetry must not break alerting.
    """
    try:
        client = sb or _supabase()
        client.table("alerts").insert({
            "type":   type_,
            "ticker": (ticker or None),
            "stage":  stage,
            "title":  title,
            "body":   body,
            "data":   data or {},
        }).execute()
    except Exception as e:
        logger.debug(f"[push] record alert failed ({type_} {ticker}): {e}")


def send_signal_alert(
    ticker: str,
    direction: str,
    confidence: int,
    signal_type: str = "stock",
    signal_id: str | None = None,
) -> None:
    """Send a new-signal push to users who have new_signals pref enabled."""
    emoji = "📈" if direction == "LONG" else "📉"
    type_label = "Options" if signal_type == "option" else "Stock"

    notif_data: dict = {"ticker": ticker, "direction": direction, "type": signal_type}
    if signal_id:
        notif_data["signal_id"] = signal_id

    title = f"{emoji} New {type_label} Signal: {ticker}"
    body  = f"{direction} · {confidence}% confidence · Tap to view details"

    # Record to the in-app feed FIRST — independent of push delivery.
    _record_alert("signal", ticker, title, body, stage=direction, data=notif_data)

    tokens = _tokens_for("new_signals")
    if not tokens:
        logger.info("[push] No tokens with new_signals enabled — skipping")
        return

    messages = [
        {
            "to":    token,
            "title": title,
            "body":  body,
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
        notif_data: dict = {"type": "t1_breakeven", "ticker": ticker, "direction": direction}
        if signal_id:
            notif_data["signal_id"] = signal_id
        title = f"🔒 {ticker} stop raised — now risk-free"
        body  = (f"We moved your stop to ${new_stop:.2f}, locking +{locked_pct:.1f}%. "
                 f"Downside is protected — update your broker stop to match.")
        _record_alert("stop", ticker, title, body, stage="breakeven", data=notif_data)

        tokens = _tokens_for("t1_breakeven")
        if not tokens:
            return
        messages = [
            {
                "to":    token,
                "title": title,
                "body":  body,
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


def send_breakdown_alert(
    ticker: str,
    stage: str,
    price: float | None = None,
    extra: str = "",
) -> int:
    """
    Broadcast a heavy-selling / breakdown alert to ALL users with the
    'breakdown_alerts' pref on (default on). NOT watchlist-scoped — it surfaces
    the strongest breakdowns across the scanned universe so a user can act even
    on names they don't already watch.

      stage="early"     → lost its 20-day average on heavy down-volume
                          (earliest structural warning — "breakdown risk")
      stage="confirmed" → broke its 20-day low on volume (breakdown confirmed)

    This is an educational RISK heads-up, not advice to short. Returns the number
    dispatched; the caller handles ranking, per-run caps and per-day dedup.
    """
    try:
        px    = f" (${price:.2f})" if price else ""
        suff  = f" · {extra}" if extra else ""
        if stage == "confirmed":
            title = f"🔻 {ticker} — Breakdown confirmed"
            body  = f"{ticker} broke its 20-day low on volume{px}{suff}. Heavy selling — breakdown risk. Tap for the read."
        else:
            title = f"⚠️ {ticker} — Heavy selling"
            body  = f"{ticker} lost its 20-day average on strong down-volume{px}{suff}. Early breakdown risk. Tap for the read."

        _record_alert("breakdown", ticker, title, body, stage=stage,
                      data={"ticker": ticker, "stage": stage})

        tokens = _tokens_for("breakdown_alerts", default=True)
        if not tokens:
            return 0
        messages = [
            {
                "to":    t,
                "title": title,
                "body":  body,
                "data":  {"type": "breakdown_alert", "ticker": ticker, "stage": stage},
                "sound": "default",
                "badge": 1,
            }
            for t in tokens
        ]
        _dispatch(messages, f"BREAKDOWN {stage} {ticker}")
        return len(messages)
    except Exception as e:
        logger.debug(f"[push] breakdown alert failed for {ticker}: {e}")
        return 0


def send_premarket_gap_alert(
    ticker: str,
    direction: str,
    strategy: str,
    gap_pct: float,
    price: float,
    stop_loss: float | None = None,
    through_stop: bool = False,
    signal_id: str | None = None,
) -> int:
    """
    Notification-ONLY premarket disaster-gap heads-up for an OPEN overnight
    position that gapped hard AGAINST the signal before the 9:30 open.

    The engine does NOT close the position or record a result on a premarket
    print — those are thin and wicky, options don't trade premarket, and the gap
    often reverses by the open. This is purely a "watch the open" warning so a
    holder isn't blindsided. Broadcast to ALL users with the 'premarket_gap_alerts'
    pref on (default on). Returns the number dispatched; the caller handles the
    per-signal-per-day dedup and the 8:00 AM ET earliest-fire gate.
    """
    try:
        is_long   = (direction or "").upper() == "LONG"
        arrow     = "📉" if is_long else "📈"
        strat_lbl = (strategy or "").replace("_", " ").strip() or "position"
        if through_stop and stop_loss:
            sl_txt = f" It's already through your ${stop_loss:.2f} stop — the open will re-price it."
        elif stop_loss:
            sl_txt = f" Your stop is ${stop_loss:.2f}."
        else:
            sl_txt = ""
        title = f"{arrow} {ticker} gapped {gap_pct:+.1f}% premarket"
        body  = (f"Your {direction} {strat_lbl} on {ticker} is moving against you "
                 f"premarket (${price:.2f}).{sl_txt} No action taken — watch the 9:30 open.")

        _record_alert("premarket_gap", ticker, title, body,
                      data={"ticker": ticker, "direction": direction,
                            "gapPct": gap_pct, "throughStop": through_stop,
                            "signal_id": signal_id})

        tokens = _tokens_for("premarket_gap_alerts", default=True)
        if not tokens:
            return 0
        data: dict = {"type": "premarket_gap", "ticker": ticker, "direction": direction}
        if signal_id:
            data["signal_id"] = signal_id
        messages = [
            {
                "to":    t,
                "title": title,
                "body":  body,
                "data":  data,
                "sound": "default",
                "badge": 1,
            }
            for t in tokens
        ]
        _dispatch(messages, f"PREMARKET-GAP {ticker} {gap_pct:+.1f}%")
        return len(messages)
    except Exception as e:
        logger.debug(f"[push] premarket gap alert failed for {ticker}: {e}")
        return 0


def send_breakout_alert(
    ticker: str,
    stage: str,
    price: float | None = None,
    extra: str = "",
) -> int:
    """
    Broadcast a breakout alert to ALL users with 'breakout_alerts' on (default on).
    Bullish mirror of send_breakdown_alert — universe-wide.

      stage="early"     → approaching its 20-day high on strong up-volume (setup forming)
      stage="confirmed" → broke its 20-day high on volume (breakout confirmed)

    Educational momentum heads-up. Returns the number dispatched; caller handles
    ranking, per-run caps and per-day dedup.
    """
    try:
        px   = f" (${price:.2f})" if price else ""
        suff = f" · {extra}" if extra else ""
        if stage == "confirmed":
            title = f"🚀 {ticker} — Breakout confirmed"
            body  = f"{ticker} broke its 20-day high on volume{px}{suff}. Momentum breakout. Tap for the read."
        else:
            title = f"⏫ {ticker} — Breakout setup"
            body  = f"{ticker} is pressing its 20-day high on strong buying{px}{suff}. A breakout may be forming. Tap for the read."

        _record_alert("breakout", ticker, title, body, stage=stage,
                      data={"ticker": ticker, "stage": stage})

        tokens = _tokens_for("breakout_alerts", default=True)
        if not tokens:
            return 0
        messages = [
            {
                "to":    t,
                "title": title,
                "body":  body,
                "data":  {"type": "breakout_alert", "ticker": ticker, "stage": stage},
                "sound": "default",
                "badge": 1,
            }
            for t in tokens
        ]
        _dispatch(messages, f"BREAKOUT {stage} {ticker}")
        return len(messages)
    except Exception as e:
        logger.debug(f"[push] breakout alert failed for {ticker}: {e}")
        return 0


def send_accumulation_alert(
    ticker: str,
    price: float | None = None,
    extra: str = "",
) -> int:
    """
    Broadcast an unusual-buying / accumulation alert to ALL users with
    'accumulation_alerts' on (default on). A lighter heads-up than a breakout —
    heavy UP-volume (big buyers stepping in) without a structural break yet.
    Returns the number dispatched; caller handles caps + per-day dedup.
    """
    try:
        px   = f" (${price:.2f})" if price else ""
        suff = f" · {extra}" if extra else ""
        title = f"🟢 {ticker} — Unusual buying"
        body  = f"{ticker} is trading on heavy up-volume{px}{suff} — big buyers may be stepping in. Tap for the read."
        _record_alert("accumulation", ticker, title, body,
                      data={"ticker": ticker})

        tokens = _tokens_for("accumulation_alerts", default=True)
        if not tokens:
            return 0
        messages = [
            {
                "to":    t,
                "title": title,
                "body":  body,
                "data":  {"type": "accumulation_alert", "ticker": ticker},
                "sound": "default",
                "badge": 1,
            }
            for t in tokens
        ]
        _dispatch(messages, f"ACCUM {ticker}")
        return len(messages)
    except Exception as e:
        logger.debug(f"[push] accumulation alert failed for {ticker}: {e}")
        return 0


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
        chg   = f" (+{change_pct:.0f}% mentions)" if change_pct is not None else ""
        title = f"🔥 {ticker} buzz spiking"
        body  = f"{ticker} is heating up on social{chg}. Tap to see why."
        _record_alert("buzz", ticker, title, body, data={"ticker": ticker}, sb=client)

        watchers = (
            client.table("watchlist").select("user_id").eq("ticker", ticker).execute().data
        ) or []
        user_ids = list({w["user_id"] for w in watchers if w.get("user_id")})
        if not user_ids:
            return 0

        tokens = _tokens_for_users(user_ids, "community_buzz")   # all devices, pref-gated
        if not tokens:
            return 0

        messages = [
            {
                "to":    t,
                "title": title,
                "body":  body,
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


def send_watchlist_state_alert(ticker: str, title: str, body: str, sb: Client | None = None) -> int:
    """
    Notify users who WATCH `ticker` that its situation changed (entered a buy
    zone, started topping, broke out, lost its trend). Watchlist-scoped, gated by
    the 'watchlist_alerts' pref (default on). Returns the number dispatched; the
    caller (watchlist_alerts.run) handles state-transition + per-day dedup.
    """
    try:
        client = sb or _supabase()
        _record_alert("watchlist", ticker, title, body, sb=client)

        watchers = (
            client.table("watchlist").select("user_id").eq("ticker", ticker).execute().data
        ) or []
        user_ids = list({w["user_id"] for w in watchers if w.get("user_id")})
        if not user_ids:
            return 0
        tokens = _tokens_for_users(user_ids, "watchlist_alerts")   # all devices, pref-gated
        if not tokens:
            return 0
        messages = [
            {
                "to":    t,
                "title": title,
                "body":  body,
                "data":  {"type": "watchlist_alert", "ticker": ticker},
                "sound": "default",
                "badge": 1,
            }
            for t in tokens
        ]
        _dispatch(messages, f"WL {ticker}")
        return len(messages)
    except Exception as e:
        logger.debug(f"[push] watchlist state alert failed for {ticker}: {e}")
        return 0


def send_commentary_alert(
    ticker: str,
    title: str,
    body: str,
    event_type: str | None = None,
    sb: Client | None = None,
) -> int:
    """
    Notify users who WATCH `ticker` of a notable LIVE intraday event on it
    (MACD cross, opening-range break, VWAP reclaim/lose, gap, sharp move) — the
    push half of the Ticker Commentary "Today's Tape". Watchlist-scoped, gated by
    the 'commentary_alerts' pref (default on). Returns the number dispatched; the
    caller (commentary_alerts.run) handles new-event detection + per-day caps so
    this never spams. Educational awareness, not advice.
    """
    try:
        client = sb or _supabase()
        _record_alert("commentary", ticker, title, body, stage=event_type,
                      data={"ticker": ticker, "event": event_type}, sb=client)

        watchers = (
            client.table("watchlist").select("user_id").eq("ticker", ticker).execute().data
        ) or []
        user_ids = list({w["user_id"] for w in watchers if w.get("user_id")})
        if not user_ids:
            return 0
        tokens = _tokens_for_users(user_ids, "commentary_alerts")   # all devices, pref-gated
        if not tokens:
            return 0
        messages = [
            {
                "to":    t,
                "title": title,
                "body":  body,
                "data":  {"type": "commentary_alert", "ticker": ticker, "event": event_type},
                "sound": "default",
                "badge": 1,
            }
            for t in tokens
        ]
        _dispatch(messages, f"CMT {ticker} {event_type or ''}")
        return len(messages)
    except Exception as e:
        logger.debug(f"[push] commentary alert failed for {ticker}: {e}")
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
        if kind == "turnaround":
            title = f"🔄 {ticker} — Turnaround Buy Zone"
            body  = f"{ticker} confirmed a reversal off the lows. Tap for the setup."
        else:
            title = f"🔻 {ticker} — Peak / Distribution"
            body  = f"{ticker} looks topped — consider taking profit / hedging. Tap for details."
        _record_alert("cycle", ticker, title, body, stage=kind,
                      data={"kind": kind, "ticker": ticker}, sb=client)

        watchers = (
            client.table("watchlist").select("user_id").eq("ticker", ticker).execute().data
        ) or []
        user_ids = list({w["user_id"] for w in watchers if w.get("user_id")})
        if not user_ids:
            return 0
        tokens = _tokens_for_users(user_ids, "cycle_signals")   # all devices, pref-gated
        if not tokens:
            return 0
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


# Terminal / result notifications that should ALSO land in the in-app Alerts feed
# (Target Hit / Stop Hit / Time Exit / Market Close / book-profit / reversal).
_FEED_RESULT_TYPES = {
    "signal_closed", "market_close", "scalp_expired", "option_expired",
    "book_profit", "reversal",
}


def _held_suffix(created_at: str | None) -> str:
    """' · opened Jun 02 · 3d held' from a signal's created_at (UTC ISO) — so a
    matured multi-day swing result doesn't read like a stale alert. '' on error."""
    if not created_at:
        return ""
    try:
        from datetime import datetime, timezone
        dt   = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        days = max(0, (datetime.now(timezone.utc) - dt).days)
        held = f"{days}d held" if days >= 1 else "same day"
        return f" · opened {dt.strftime('%b %d')} · {held}"
    except Exception:
        return ""


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

    # Terminal results: append a hold-duration suffix (so a matured multi-day
    # swing doesn't read as stale) and record to the in-app Alerts feed — both
    # independent of push delivery, so closes show in the Alerts tab too.
    if notif_type in _FEED_RESULT_TYPES:
        suffix = _held_suffix(payload.get("created_at"))
        if suffix and suffix not in (body or ""):
            body = f"{body}{suffix}"
        try:
            _record_alert("result", payload.get("ticker"), title, body,
                          stage=payload.get("result") or notif_type, data=payload)
        except Exception:
            pass

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


# Admin-only ops alerts (internal monitoring — NEVER broadcast to users).
ADMIN_EMAIL = "techsoftmgr@gmail.com"


def send_admin_alert(title: str, body: str, data: dict | None = None) -> bool:
    """
    Push to the ADMIN's device(s) ONLY (never a broadcast). For internal ops
    alerts such as the daily phantom-data audit. Now multi-device: sends to ALL
    of the admin's registered devices. Returns True if a token was found and
    dispatched (False = logged only, e.g. no token registered).
    """
    try:
        tokens = [d["token"] for d in _device_rows() if d.get("email") == ADMIN_EMAIL]
        if not tokens:
            logger.info("[push] admin alert: no admin push token registered — logged only")
            return False
        messages = [
            {"to": t, "title": title, "body": body,
             "data": {**(data or {}), "type": "admin_ops"},
             "sound": "default", "badge": 1}
            for t in tokens
        ]
        _dispatch(messages, title)
        return True
    except Exception as e:
        logger.warning(f"[push] admin alert failed: {e}")
        return False


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
        data = result.get("data", []) or []
        errors = [
            r.get("details", {}).get("error")
            for r in data
            if r.get("status") == "error"
        ]
        # Prune tokens Expo says are dead (DeviceNotRegistered) — the response is
        # positional, so index i maps to messages[i]["to"].
        dead = [
            messages[i].get("to")
            for i, r in enumerate(data)
            if i < len(messages)
            and r.get("status") == "error"
            and r.get("details", {}).get("error") == "DeviceNotRegistered"
        ]
        if dead:
            _prune_dead_tokens(dead)
        if errors:
            logger.warning(f"[push] Errors for '{label}': {errors}")
        else:
            logger.info(f"[push] Sent {len(messages)} notification(s): {label}")
    except Exception as e:
        logger.error(f"[push] Dispatch failed for '{label}': {e}")
