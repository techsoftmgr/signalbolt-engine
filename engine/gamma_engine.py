"""
Gamma Exposure Engine
=====================
Fetches SpotGamma data to identify:
  - Gamma walls (where MMs must hedge = price resistance/support)
  - GEX flip level (where dealer gamma changes sign)
  - Max pain strike (where options sellers profit most = price pin target)
  - Vanna / Charm flow direction
  - Pin risk on OpEx

Critical for:
  - SL placement: beyond gamma level, not at obvious stop cluster
  - TP capping: before gamma wall (MMs dump at wall)
  - Regime context: negative GEX = amplified moves

Falls back gracefully if SpotGamma key not configured.
"""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger("signalbolt.gamma")

SPOTGAMMA_BASE = "https://api.spotgamma.com/v1"
WALL_THRESHOLD_MILLIONS = 50   # GEX > $50M = significant wall
PIN_RISK_DISTANCE = 0.005      # within 0.5% of max pain = pin risk


def _api_key() -> Optional[str]:
    return os.environ.get("SPOTGAMMA_API_KEY")


def _get(endpoint: str, params: dict = None) -> Optional[dict]:
    key = _api_key()
    if not key:
        return None
    try:
        headers = {"x-api-key": key}
        r = requests.get(
            f"{SPOTGAMMA_BASE}/{endpoint}",
            params=params or {},
            headers=headers,
            timeout=5,
        )
        if r.status_code == 200:
            return r.json()
        logger.debug(f"[gamma] SpotGamma {endpoint} → {r.status_code}")
        return None
    except Exception as e:
        logger.debug(f"[gamma] request error: {e}")
        return None


def _find_nearest_walls(strikes: list, current_price: float) -> dict:
    """Find nearest significant gamma wall above and below current price."""
    above = [s for s in strikes if s["strike"] > current_price and abs(s["gex"]) > WALL_THRESHOLD_MILLIONS]
    below = [s for s in strikes if s["strike"] < current_price and abs(s["gex"]) > WALL_THRESHOLD_MILLIONS]

    above_sorted = sorted(above, key=lambda x: x["strike"])
    below_sorted = sorted(below, key=lambda x: x["strike"], reverse=True)

    def to_wall(s: dict, side: str) -> dict:
        return {
            "strike":       s["strike"],
            "gex":          s["gex"],
            "distance_pct": abs(s["strike"] - current_price) / current_price,
            "side":         side,
        }

    return {
        "wall_above": to_wall(above_sorted[0], "ABOVE") if above_sorted else None,
        "wall_below": to_wall(below_sorted[0], "BELOW") if below_sorted else None,
    }


def fetch(ticker: str, current_price: float) -> dict:
    """
    Fetch gamma exposure data for ticker.

    Returns:
        {
          "ticker":           str,
          "net_gex":          float,   # $ millions, positive = long gamma
          "is_negative_gamma": bool,   # MMs amplify moves when True
          "flip_level":       float,   # GEX zero-crossing strike
          "max_pain":         float,   # max pain strike
          "wall_above":       dict|None,
          "wall_below":       dict|None,
          "vanna_tailwind":   bool,    # vanna flow helps signal direction
          "charm_headwind":   bool,    # charm flow hurts signal (Friday PM)
          "pin_risk":         bool,
          "available":        bool,    # False if SpotGamma not configured
          "score":            float,   # 0-100 quality score
        }
    """
    # Default / fallback
    fallback = {
        "ticker":            ticker,
        "net_gex":           0.0,
        "is_negative_gamma": False,
        "flip_level":        0.0,
        "max_pain":          current_price,
        "wall_above":        None,
        "wall_below":        None,
        "vanna_tailwind":    False,
        "charm_headwind":    False,
        "pin_risk":          False,
        "available":         False,
        "score":             60.0,   # neutral when data unavailable
    }

    if not _api_key():
        return fallback

    gex_data   = _get("gex", {"ticker": ticker})
    level_data = _get("levels", {"ticker": ticker})

    if not gex_data or not level_data:
        return fallback

    try:
        data        = gex_data.get("data", {})
        net_gex     = float(data.get("net_gex_millions", 0))
        flip_level  = float(level_data.get("data", {}).get("flip_level", 0))
        max_pain    = float(level_data.get("data", {}).get("max_pain", current_price))
        vanna_raw   = float(data.get("vanna_exposure", 0))
        charm_raw   = float(data.get("charm_exposure", 0))

        strikes_raw = data.get("strikes", [])
        strikes = [
            {"strike": float(s["strike"]), "gex": float(s.get("gex_millions", 0))}
            for s in strikes_raw
        ]

        walls = _find_nearest_walls(strikes, current_price)
        pin_risk = abs(current_price - max_pain) / current_price < PIN_RISK_DISTANCE

        result = {
            "ticker":            ticker,
            "net_gex":           round(net_gex, 1),
            "is_negative_gamma": net_gex < 0,
            "flip_level":        round(flip_level, 2),
            "max_pain":          round(max_pain, 2),
            "wall_above":        walls["wall_above"],
            "wall_below":        walls["wall_below"],
            "vanna_tailwind":    vanna_raw > 0.1,
            "charm_headwind":    charm_raw < -0.1,
            "pin_risk":          pin_risk,
            "available":         True,
            "score":             0.0,   # computed below
        }

        result["score"] = _compute_score(result)

        logger.info(
            f"[gamma] {ticker} GEX={net_gex:.0f}M "
            f"{'NEG' if net_gex < 0 else 'POS'} | "
            f"wall↑={walls['wall_above']['strike'] if walls['wall_above'] else 'none'} "
            f"wall↓={walls['wall_below']['strike'] if walls['wall_below'] else 'none'} "
            f"| maxpain={max_pain:.0f} pin={'Y' if pin_risk else 'N'}"
        )
        return result

    except Exception as e:
        logger.debug(f"[gamma] parse error for {ticker}: {e}")
        return fallback


def _compute_score(g: dict) -> float:
    score = 70.0
    if g["is_negative_gamma"]:  score -= 20  # amplified moves = SL risk
    else:                        score += 8   # MMs absorb = stable moves
    if g["pin_risk"]:            score -= 20  # price pinned at expiry
    if g["wall_above"] and g["wall_above"]["distance_pct"] < 0.005:
        score -= 15   # gamma wall extremely close above (TP blocked)
    if g["vanna_tailwind"]:      score += 8
    if g["charm_headwind"]:      score -= 8
    return max(0.0, min(100.0, score))


def score_for_signal(gamma: dict, direction: str, is_opex_day: bool) -> float:
    """
    Return 0-100 score for gamma conditions given signal direction.
    Used as L8 bonus in scorer.py.
    """
    if not gamma.get("available"):
        return 65.0   # neutral when data not available

    base = gamma.get("score", 65.0)

    # Direction-specific wall check
    if direction == "LONG":
        wall = gamma.get("wall_above")
        if wall and wall["distance_pct"] < 0.005:
            base -= 20   # wall within 0.5% = TP will be immediately blocked
        elif wall and wall["distance_pct"] > 0.02:
            base += 5    # wall far away = room to run
        if gamma.get("vanna_tailwind"): base += 5
        if gamma.get("charm_headwind"): base -= 5

    elif direction == "SHORT":
        wall = gamma.get("wall_below")
        if wall and wall["distance_pct"] < 0.005:
            base -= 20
        elif wall and wall["distance_pct"] > 0.02:
            base += 5
        if gamma.get("charm_headwind"): base += 5

    if gamma.get("pin_risk") and is_opex_day:
        base -= 20

    return max(0.0, min(100.0, base))


def adjust_sl_for_gamma(sl: float, direction: str, gamma: dict, entry: float) -> tuple[float, str]:
    """
    Shift SL beyond the nearest gamma level to avoid stop raids at obvious levels.
    Returns (adjusted_sl, reason_string).
    """
    if not gamma.get("available"):
        return sl, ""

    if direction == "LONG":
        wall = gamma.get("wall_below")
        if wall and sl > wall["strike"] * 0.995:
            new_sl = round(wall["strike"] * 0.992, 2)
            return new_sl, f"SL moved below gamma support ${wall['strike']:.2f}"

    elif direction == "SHORT":
        wall = gamma.get("wall_above")
        if wall and sl < wall["strike"] * 1.005:
            new_sl = round(wall["strike"] * 1.008, 2)
            return new_sl, f"SL moved above gamma resistance ${wall['strike']:.2f}"

    return sl, ""


def adjust_tp_for_gamma_wall(tp: float, direction: str, gamma: dict) -> tuple[float, str]:
    """
    Cap TP before gamma wall — MMs suppress/dump at wall.
    Returns (adjusted_tp, reason_string).
    """
    if not gamma.get("available"):
        return tp, ""

    if direction == "LONG":
        wall = gamma.get("wall_above")
        if wall and tp >= wall["strike"]:
            new_tp = round(wall["strike"] * 0.992, 2)
            return new_tp, f"TP capped below gamma wall ${wall['strike']:.2f}"

    elif direction == "SHORT":
        wall = gamma.get("wall_below")
        if wall and tp <= wall["strike"]:
            new_tp = round(wall["strike"] * 1.008, 2)
            return new_tp, f"TP adjusted above gamma support ${wall['strike']:.2f}"

    return tp, ""
