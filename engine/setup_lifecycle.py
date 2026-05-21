"""
Setup Lifecycle Manager
=======================
Manages the progression of setups through confidence stages:

    WATCHLIST → DEVELOPING → CONFIRMED_SIGNAL → (fired to signals table)
                                              ↘ EXPIRED / INVALIDATED

Why this exists:
  A setup scoring 62 isn't garbage. It might be missing ONE confirmation
  (HTF alignment, volume expansion, or regime shift). Discarding it means
  losing the opportunity when it matures. Instead, track it, re-evaluate
  every scan cycle, and fire the signal only when it reaches full threshold.

State transitions:
  WATCHLIST   (50–64)  → re-evaluated each scan → may become DEVELOPING
  DEVELOPING  (65–77)  → re-evaluated each scan → may become CONFIRMED or expire
  CONFIRMED   (78+)    → promoted to signals table, push notification sent
  EXPIRED              → structure invalidated, setup aged out, or regime changed
  INVALIDATED          → explicit structure failure detected mid-lifecycle

DB table: setup_watchlist (separate from signals table)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

from supabase import Client

logger = logging.getLogger("signalbolt.lifecycle")


# ── Enums ─────────────────────────────────────────────────────────────────────

class SetupState(str, Enum):
    WATCHLIST        = "WATCHLIST"
    DEVELOPING       = "DEVELOPING"
    CONFIRMED_SIGNAL = "CONFIRMED_SIGNAL"
    EXPIRED          = "EXPIRED"
    INVALIDATED      = "INVALIDATED"


class ConfidenceGrade(str, Enum):
    """
    Replaces raw 0–100 numeric confidence with a qualitative grade.
    Communicates quality without implying false precision.
    """
    A_PLUS = "A+"   # ≥ 90
    A      = "A"    # ≥ 82
    B_PLUS = "B+"   # ≥ 74
    B      = "B"    # ≥ 66
    C      = "C"    # < 66


class RiskGrade(str, Enum):
    """Overall trade risk assessment."""
    LOW    = "LOW"     # tight SL, clean structure, good regime
    MEDIUM = "MEDIUM"  # moderate SL, some uncertainty
    HIGH   = "HIGH"    # wide SL, conflicting signals, or volatile regime


# ── Score band thresholds ──────────────────────────────────────────────────────
WATCHLIST_MIN        = 50
DEVELOPING_MIN       = 65
CONFIRMED_MIN        = 78

# Setup expiry: if no improvement in N scans, expire it
WATCHLIST_MAX_AGE_HOURS   = 2.0    # watchlist setups expire in 2h if stuck
DEVELOPING_MAX_AGE_HOURS  = 1.0    # developing setups expire in 1h if stuck
MAX_SCANS_WITHOUT_IMPROVE = 8      # max re-evaluations before forced expiry


# ── Grade classifiers ─────────────────────────────────────────────────────────

def classify_state(score: float) -> SetupState:
    if score >= CONFIRMED_MIN:
        return SetupState.CONFIRMED_SIGNAL
    if score >= DEVELOPING_MIN:
        return SetupState.DEVELOPING
    if score >= WATCHLIST_MIN:
        return SetupState.WATCHLIST
    return SetupState.EXPIRED


def classify_confidence_grade(score: float) -> ConfidenceGrade:
    if score >= 90:
        return ConfidenceGrade.A_PLUS
    if score >= 82:
        return ConfidenceGrade.A
    if score >= 74:
        return ConfidenceGrade.B_PLUS
    if score >= 66:
        return ConfidenceGrade.B
    return ConfidenceGrade.C


def classify_risk_grade(
    score:       float,
    risk_reward: float,
    chop_score:  float,
    regime_type: str,
) -> RiskGrade:
    """Assess trade risk based on score, R:R, chop, and regime."""
    risk_points = 0

    # Score below B+ → higher risk
    if score < 74:
        risk_points += 2
    elif score < 82:
        risk_points += 1

    # R:R below 1.5 → risky
    if risk_reward < 1.5:
        risk_points += 2
    elif risk_reward < 2.0:
        risk_points += 1

    # Chop score elevated
    if chop_score > 50:
        risk_points += 2
    elif chop_score > 35:
        risk_points += 1

    # Hostile regime
    if regime_type in ("PANIC", "HIGH_VOL"):
        risk_points += 2
    elif regime_type in ("RANGING", "RISK_OFF"):
        risk_points += 1

    if risk_points >= 4:
        return RiskGrade.HIGH
    if risk_points >= 2:
        return RiskGrade.MEDIUM
    return RiskGrade.LOW


# ── Missing confirmation detector ─────────────────────────────────────────────

def get_missing_confirmations(
    analysis:    dict,
    score_result: dict,
    regime:      dict,
    chop,                   # ChopResult object OR dict OR None
) -> list[str]:
    """
    Returns a human-readable list of what confirmations the setup is missing
    to reach CONFIRMED_SIGNAL threshold. Shown to users in the "Developing Setup" card.

    chop accepts: ChopResult dataclass, plain dict, or None — all handled safely.
    """
    missing: list[str] = []

    breakdown   = score_result.get("breakdown", {})
    total       = score_result.get("total", 0)
    direction   = analysis.get("direction", "")
    regime_type = (regime or {}).get("regime_type", "UNKNOWN")

    # ── Safely extract chop fields from any input type ─────────
    if chop is None:
        chop_vol_ratio = 1.0
        chop_score_val = 0.0
    elif isinstance(chop, dict):
        chop_vol_ratio = chop.get("vol_ratio", 1.0)
        chop_score_val = chop.get("chop_score", 0.0)
    else:
        # ChopResult dataclass
        chop_vol_ratio = getattr(chop, "vol_ratio", 1.0)
        chop_score_val = getattr(chop, "chop_score", 0.0)

    # Structure: L1 weak
    l1 = breakdown.get("l1_smc", 0)
    if l1 < 12:
        missing.append("Confirmed BOS or CHoCH structure required")
    elif l1 < 18:
        missing.append("Stronger structure confluence (OB or FVG at setup zone)")

    # Technical: L2 weak
    l2 = breakdown.get("l2_technical", 0)
    if l2 < 14:
        missing.append("Technical alignment (RSI + MACD + VWAP agreement)")

    # MTF: L5 not aligned
    l5 = breakdown.get("l5_mtf", 0)
    if l5 < 7.5:
        missing.append("Higher-timeframe (4H) structure alignment")

    # Volume
    if chop_vol_ratio < 0.80:
        missing.append("Volume expansion (current volume below average)")

    # Regime
    if regime_type in ("PANIC", "HIGH_VOL"):
        missing.append(f"Regime improvement (currently {regime_type} — high risk)")
    elif regime_type == "RANGING" and direction in ("LONG", "SHORT"):
        missing.append("Ranging market — wait for breakout or use mean-reversion setup")

    # Chop
    if chop_score_val > 50:
        missing.append(f"Cleaner price action (chop score={chop_score_val:.0f}/100)")

    # Score gap
    gap = CONFIRMED_MIN - total
    if gap > 10:
        missing.append(f"Confidence score {total} → need {CONFIRMED_MIN} ({gap} pts gap)")

    return missing[:5]  # cap at 5 items to keep UI clean


# ── Setup type classifier ─────────────────────────────────────────────────────

SETUP_TYPES = {
    "CHOCH_OB_RETEST":           "CHoCH confirmed + OB retest entry",
    "BOS_CONTINUATION":          "Break of Structure continuation",
    "FVG_RETEST":                "Fair Value Gap retest",
    "LIQUIDITY_SWEEP_REVERSAL":  "Liquidity sweep + structural reversal",
    "VWAP_MEAN_REVERSION":       "VWAP deviation mean-reversion",
    "ORB_BREAKOUT":              "Opening Range Breakout",
    "OPTIONS_FLOW_CONFIRMATION": "Unusual options flow + price confirmation",
    "DARK_POOL_ACCUMULATION":    "Dark pool block + trend alignment",
}


def classify_setup_type(analysis: dict, session: dict) -> str:
    """
    Classify the setup into one of the canonical setup types.
    These are stored per-signal for analytics and optimizer feedback.
    """
    structure  = analysis.get("structure", {})
    sweep      = analysis.get("liquidity_sweep", {})
    fvgs       = analysis.get("fvgs", {})
    obs        = analysis.get("obs",  {})
    direction  = analysis.get("direction", "LONG")
    strategy   = analysis.get("strategy_type", "day_trade")
    session_mode = (session or {}).get("mode", "STANDARD")

    is_long = direction == "LONG"

    # Priority order: most specific → most general
    if strategy == "options_flow":
        return "OPTIONS_FLOW_CONFIRMATION"

    if strategy == "dark_pool":
        return "DARK_POOL_ACCUMULATION"

    if session_mode == "ORB":
        return "ORB_BREAKOUT"

    if strategy == "mean_reversion":
        return "VWAP_MEAN_REVERSION"

    # Sweep reversal: sweep present + CHoCH
    if sweep.get("swept") and (
        (is_long  and structure.get("choch_bullish")) or
        (not is_long and structure.get("choch_bearish"))
    ):
        return "LIQUIDITY_SWEEP_REVERSAL"

    # CHoCH + OB
    if (
        (is_long  and structure.get("choch_bullish") and obs.get("ob_bullish")) or
        (not is_long and structure.get("choch_bearish") and obs.get("ob_bearish"))
    ):
        return "CHOCH_OB_RETEST"

    # BOS continuation
    if (
        (is_long  and structure.get("bos_bullish")  and not structure.get("choch_bullish")) or
        (not is_long and structure.get("bos_bearish") and not structure.get("choch_bearish"))
    ):
        return "BOS_CONTINUATION"

    # FVG retest
    fvg_key = "fvg_bullish" if is_long else "fvg_bearish"
    if fvgs.get(fvg_key):
        return "FVG_RETEST"

    return "BOS_CONTINUATION"  # default fallback


# ── Supabase integration ──────────────────────────────────────────────────────

class SetupLifecycleManager:
    """
    Manages WATCHLIST/DEVELOPING setups in the setup_watchlist table.
    Promoted setups get written to the main signals table by the runner.

    Supabase client is initialised lazily on first use so the class can be
    instantiated at module-level without requiring env vars at import time.
    """

    def __init__(self, sb: Optional[Client] = None):
        self._sb_provided = sb   # optional pre-supplied client (tests / DI)
        self._sb_lazy: Optional[Client] = None

    @property
    def sb(self) -> Client:
        """Return a live Supabase client, creating one lazily if needed."""
        if self._sb_provided is not None:
            return self._sb_provided
        if self._sb_lazy is None:
            try:
                from supabase import create_client
                url = os.environ.get("SUPABASE_URL", "")
                key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "")
                if not url or not key:
                    raise ValueError("SUPABASE_URL / SUPABASE_SERVICE_KEY not set")
                self._sb_lazy = create_client(url, key)
            except Exception as exc:
                raise RuntimeError(f"SetupLifecycleManager: cannot connect to Supabase: {exc}") from exc
        return self._sb_lazy

    def upsert_setup(
        self,
        analysis:     dict,
        score_result: dict,
        regime:       dict,
        session:      dict,
        chop_result:  dict,
        setup_type:   str,
        sltp:         dict,
    ) -> Optional[str]:
        """
        Save or update a WATCHLIST/DEVELOPING setup.
        Returns the setup_id if saved, None if below WATCHLIST_MIN.

        If setup already exists for this ticker+direction+strategy, update it.
        """
        score     = score_result.get("total", 0)
        state     = classify_state(score)

        if state == SetupState.EXPIRED:
            return None   # below 50 — not worth tracking

        if state == SetupState.CONFIRMED_SIGNAL:
            return None   # confirmed signals go directly to signals table

        ticker     = analysis.get("ticker", "")
        direction  = analysis.get("direction", "")
        strategy   = analysis.get("strategy_type", "day_trade")
        breakdown  = score_result.get("breakdown", {})
        regime_t   = (regime or {}).get("regime_type", "")
        # Handle ChopResult dataclass or dict or None
        if chop_result is None:
            chop_score = 0
        elif isinstance(chop_result, dict):
            chop_score = chop_result.get("chop_score", 0)
        else:
            chop_score = getattr(chop_result, "chop_score", 0)

        # Compute missing confirmations (handles ChopResult, dict, or None)
        missing = get_missing_confirmations(analysis, score_result, regime or {}, chop_result)

        # Determine grades
        conf_grade = classify_confidence_grade(score)
        risk_grade = classify_risk_grade(
            score=score,
            risk_reward=sltp.get("risk_reward_1", 0) if sltp else 0,
            chop_score=chop_score,
            regime_type=regime_t,
        )

        # Confirmation trigger for UI
        trigger = _build_confirmation_trigger(analysis, missing)

        # Invalidation level
        direction_val = analysis.get("direction", "LONG")
        entry = analysis.get("entry") or analysis.get("current_price")
        sl    = sltp.get("stop_loss") if sltp else None
        invalidation = sl if sl else (
            round(entry * 0.98, 2) if entry and direction_val == "LONG"
            else round(entry * 1.02, 2) if entry else None
        )

        # Expiry time based on state
        now = datetime.now(timezone.utc)
        if state == SetupState.WATCHLIST:
            expires_at = (now + timedelta(hours=WATCHLIST_MAX_AGE_HOURS)).isoformat()
        else:
            expires_at = (now + timedelta(hours=DEVELOPING_MAX_AGE_HOURS)).isoformat()

        payload = {
            "ticker":                ticker,
            "direction":             direction,
            "strategy_type":         strategy,
            "setup_type":            setup_type,
            "setup_state":           state.value,
            "confidence_grade":      conf_grade.value,
            "risk_grade":            risk_grade.value,
            "score":                 score,
            "score_breakdown":       breakdown,
            "missing_confirmations": missing,
            "entry_zone_top":        round(float(entry) * 1.002, 4) if entry else None,
            "entry_zone_bot":        round(float(entry) * 0.998, 4) if entry else None,
            "stop_loss":             sltp.get("stop_loss") if sltp else None,
            "target_one":            sltp.get("target_one") if sltp else None,
            "target_two":            sltp.get("target_two") if sltp else None,
            "invalidation_level":    invalidation,
            "confirmation_trigger":  trigger,
            "regime_type":           regime_t,
            "regime_alignment":      _regime_alignment(regime_t, direction),
            "catalyst_present":      score_result.get("breakdown", {}).get("sweep_confirmed", False),
            "chop_score":            chop_score,
            "expires_at":            expires_at,
            "updated_at":            now.isoformat(),
        }

        try:
            # Check if setup already exists (same ticker + direction + strategy, active)
            existing = (
                self.sb.table("setup_watchlist")
                .select("id, score, scan_count, setup_state")
                .eq("ticker", ticker)
                .eq("direction", direction)
                .eq("strategy_type", strategy)
                .in_("setup_state", [SetupState.WATCHLIST.value, SetupState.DEVELOPING.value])
                .execute()
                .data
            )

            if existing:
                row = existing[0]
                setup_id  = row["id"]
                new_count = (row.get("scan_count") or 1) + 1

                # Force expiry if no improvement after MAX_SCANS scans
                if new_count > MAX_SCANS_WITHOUT_IMPROVE and score <= row.get("score", 0):
                    self._expire_setup(setup_id, "No score improvement after max scans")
                    return None

                payload["scan_count"] = new_count
                self.sb.table("setup_watchlist").update(payload).eq("id", setup_id).execute()
                logger.info(
                    f"[lifecycle] Updated {state.value} setup {ticker} {direction} "
                    f"score={score} scan#{new_count} grade={conf_grade.value}"
                )
                return setup_id
            else:
                payload["scan_count"] = 1
                result = self.sb.table("setup_watchlist").insert(payload).execute()
                setup_id = result.data[0]["id"] if result.data else None
                logger.info(
                    f"[lifecycle] NEW {state.value} setup {ticker} {direction} "
                    f"score={score} grade={conf_grade.value} missing={missing}"
                )
                return setup_id

        except Exception as e:
            logger.warning(f"[lifecycle] Failed to save setup for {ticker}: {e}")
            return None

    def expire_stale_setups(self) -> int:
        """
        Expire watchlist setups that have passed their expiry time.
        Called at the start of each scan cycle.
        Returns count of setups expired.
        """
        try:
            now = datetime.now(timezone.utc).isoformat()
            result = (
                self.sb.table("setup_watchlist")
                .update({
                    "setup_state": SetupState.EXPIRED.value,
                    "updated_at":  now,
                })
                .in_("setup_state", [SetupState.WATCHLIST.value, SetupState.DEVELOPING.value])
                .lt("expires_at", now)
                .execute()
            )
            count = len(result.data) if result.data else 0
            if count > 0:
                logger.info(f"[lifecycle] Expired {count} stale watchlist setups")
            return count
        except Exception as e:
            logger.warning(f"[lifecycle] Expire stale failed: {e}")
            return 0

    def invalidate_setup(self, ticker: str, direction: str, strategy: str, reason: str) -> None:
        """Mark a specific setup as INVALIDATED when structure fails."""
        try:
            self.sb.table("setup_watchlist").update({
                "setup_state": SetupState.INVALIDATED.value,
                "updated_at":  datetime.now(timezone.utc).isoformat(),
                "confirmation_trigger": f"INVALIDATED: {reason}",
            }).eq("ticker", ticker).eq("direction", direction).eq(
                "strategy_type", strategy
            ).in_(
                "setup_state", [SetupState.WATCHLIST.value, SetupState.DEVELOPING.value]
            ).execute()
            logger.info(f"[lifecycle] Invalidated {ticker} {direction} — {reason}")
        except Exception as e:
            logger.debug(f"[lifecycle] Invalidate failed: {e}")

    def mark_promoted(self, ticker: str, direction: str, strategy: str, signal_id: str) -> None:
        """Mark a watchlist setup as promoted to a confirmed signal."""
        try:
            self.sb.table("setup_watchlist").update({
                "setup_state":          SetupState.CONFIRMED_SIGNAL.value,
                "promoted_to_signal_id": signal_id,
                "updated_at":           datetime.now(timezone.utc).isoformat(),
            }).eq("ticker", ticker).eq("direction", direction).eq(
                "strategy_type", strategy
            ).in_(
                "setup_state", [SetupState.WATCHLIST.value, SetupState.DEVELOPING.value]
            ).execute()
        except Exception as e:
            logger.debug(f"[lifecycle] mark_promoted failed: {e}")

    def _expire_setup(self, setup_id: str, reason: str) -> None:
        try:
            self.sb.table("setup_watchlist").update({
                "setup_state":          SetupState.EXPIRED.value,
                "confirmation_trigger": f"EXPIRED: {reason}",
                "updated_at":           datetime.now(timezone.utc).isoformat(),
            }).eq("id", setup_id).execute()
        except Exception:
            pass

    def get_active_watchlist(self, ticker: Optional[str] = None) -> list[dict]:
        """Get all active WATCHLIST/DEVELOPING setups, optionally filtered by ticker."""
        try:
            query = (
                self.sb.table("setup_watchlist")
                .select("*")
                .in_("setup_state", [SetupState.WATCHLIST.value, SetupState.DEVELOPING.value])
            )
            if ticker:
                query = query.eq("ticker", ticker)
            return query.execute().data or []
        except Exception as e:
            logger.debug(f"[lifecycle] get_active_watchlist failed: {e}")
            return []


# ── Helper functions ──────────────────────────────────────────────────────────

def _build_confirmation_trigger(analysis: dict, missing: list[str]) -> str:
    """Build a human-readable confirmation trigger description."""
    if not missing:
        return "All confirmations present"

    # Take the first missing confirmation and frame it as a trigger
    first = missing[0].lower()
    if "bos" in first or "choch" in first or "structure" in first:
        return "Wait for confirmed BOS/CHoCH break with 2 closes above/below level"
    if "volume" in first:
        return "Wait for volume expansion above 20-bar average"
    if "timeframe" in first or "4h" in first:
        return "Wait for 4H chart structure to align with direction"
    if "regime" in first:
        return f"Wait for regime improvement (currently {missing[0].split('(')[-1].rstrip(')')}"
    if "chop" in first:
        return "Wait for price action to stabilize (reduce chop score)"
    return f"Requires: {missing[0]}"


def _regime_alignment(regime_type: str, direction: str) -> str:
    """Score regime alignment with signal direction."""
    bull_friendly = {"TRENDING_BULL", "LOW_VOL"}
    bear_friendly = {"TRENDING_BEAR", "RISK_OFF"}
    neutral       = {"RANGING", "HIGH_VOL"}
    hostile       = {"PANIC"}

    if regime_type in hostile:
        return "HOSTILE"
    if direction == "LONG" and regime_type in bull_friendly:
        return "ALIGNED"
    if direction == "SHORT" and regime_type in bear_friendly:
        return "ALIGNED"
    if regime_type in neutral:
        return "NEUTRAL"
    return "OPPOSED"


# ── Score annotation helpers (for API response) ───────────────────────────────

def annotate_score(
    score:       float,
    breakdown:   dict,
    direction:   str,
    regime_type: str,
) -> dict:
    """
    Return human-readable score annotation for API response.
    Replaces raw 100-point score with grade + explanation.
    """
    grade         = classify_confidence_grade(score)
    state         = classify_state(score)
    regime_align  = _regime_alignment(regime_type, direction)

    l1 = breakdown.get("l1_smc", 0)
    l2 = breakdown.get("l2_technical", 0)
    l3 = breakdown.get("l3_sentiment", 0)
    l6 = breakdown.get("l6_regime", 0)

    strongest_layer  = max(breakdown, key=lambda k: breakdown.get(k, 0) if isinstance(breakdown.get(k), (int, float)) else 0)
    weakest_layer    = min(
        (k for k in ["l1_smc", "l2_technical", "l3_sentiment", "l4_risk", "l5_mtf"]),
        key=lambda k: breakdown.get(k, 0),
    )

    layer_names = {
        "l1_smc": "SMC Structure",
        "l2_technical": "Technical Indicators",
        "l3_sentiment": "Sentiment/Flow",
        "l4_risk": "Risk Environment",
        "l5_mtf": "Multi-Timeframe",
    }

    return {
        "score":           score,
        "confidence_grade": grade.value,
        "setup_state":     state.value,
        "regime_alignment": regime_align,
        "score_explanation": (
            f"{grade.value} setup — strongest: {layer_names.get(strongest_layer, strongest_layer)}, "
            f"weakest: {layer_names.get(weakest_layer, weakest_layer)}"
        ),
        "confidence_reason": _confidence_reason(grade, l1, l2, l3),
    }


def _confidence_reason(grade: ConfidenceGrade, l1: float, l2: float, l3: float) -> str:
    if grade == ConfidenceGrade.A_PLUS:
        return "Exceptional setup — all layers strongly aligned"
    if grade == ConfidenceGrade.A:
        return "High-quality setup — most layers aligned"
    if grade == ConfidenceGrade.B_PLUS:
        return "Solid setup with minor gaps — within acceptable range"
    if grade == ConfidenceGrade.B:
        return "Moderate setup — proceed with standard sizing"
    return "Below-par setup — risk of false positive, reduce size or skip"
