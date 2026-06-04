"""
Options signal scanner — deep analysis before firing any contract.

Data sources:
  1. Polygon.io options snapshot (primary) — real Greeks, accurate IV
  2. yfinance options chain (fallback) — if Polygon unavailable or no results

Filters applied (any failure = skip):
  1. Earnings proximity   — skip if earnings within 5 days
  2. Expiry window        — 14-30 DTE (≈2-4 weeks): long enough to survive a
                            1-10 day swing hold, short enough to stay responsive
                            and limit theta. (NOT daily/0DTE — it would expire
                            before a multi-day swing resolves.)
  3. Strike               — slightly IN-the-money (~2%, delta ~0.6) so the premium
                            tracks the underlying move ~1:1 with less extrinsic
                            value to decay than an ATM/OTM contract.
  4. Liquidity gate       — OI >= 500, premium >= $0.10
  5. Flow validation      — volume must not exceed OI (closers vs openers)
  6. IV vs HV check       — skip if IV > 1.5× 30-day realised vol (overpriced)
"""

import logging
import math
import os
import requests as _requests
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

POLYGON_KEY    = os.environ.get("POLYGON_API_KEY", "")
# Expiry window matched to the 1-10 day swing hold: ~2-4 weeks. Long enough to
# survive the full hold + a buffer, short enough to track the move and limit the
# theta you pay (was 21-60, which overpaid for time and tracked the swing less).
_MIN_DTE       = 14
_MAX_DTE       = 30
# Target ~2% IN-the-money (delta ~0.6): premium tracks the underlying move closer
# to 1:1 with less extrinsic value to decay than ATM/OTM. For a CALL that's just
# BELOW spot; for a PUT just ABOVE spot.
_ITM_OFFSET    = 0.02
_RISK_FREE     = 0.05
_MIN_OI        = 500
_MIN_ASK       = 0.10
_MAX_EARN_DAYS = 5
_IV_HV_MAX_RATIO = 1.5


def _target_strike(current_price: float, is_call: bool) -> float:
    """Slightly IN-the-money target strike (delta ~0.6). CALL → just below spot,
    PUT → just above spot, so the contract carries intrinsic value and tracks the
    underlying move more 1:1 (less extrinsic/theta than ATM or OTM)."""
    return current_price * ((1 - _ITM_OFFSET) if is_call else (1 + _ITM_OFFSET))


# ---------------------------------------------------------------------------
# Black-Scholes helpers (used when Polygon doesn't return Greeks)
# ---------------------------------------------------------------------------

def _ncdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

def _bs_delta(S, K, T, r, sigma, is_call):
    if T <= 0 or sigma <= 0:
        return 1.0 if (is_call and S > K) else 0.0
    d = _d1(S, K, T, r, sigma)
    return round(_ncdf(d) if is_call else _ncdf(d) - 1.0, 3)

def _bs_theta(S, K, T, r, sigma, is_call):
    if T <= 0 or sigma <= 0:
        return 0.0
    d  = _d1(S, K, T, r, sigma)
    d2 = d - sigma * math.sqrt(T)
    t1 = -(S * _npdf(d) * sigma) / (2.0 * math.sqrt(T))
    t2 = (-r * K * math.exp(-r * T) * _ncdf(d2)) if is_call else (r * K * math.exp(-r * T) * _ncdf(-d2))
    return round((t1 + t2) / 365.0, 4)


# ---------------------------------------------------------------------------
# Shared filter helpers
# ---------------------------------------------------------------------------

def _dte(expiry_str: str) -> int:
    return (datetime.strptime(expiry_str, "%Y-%m-%d").date() - date.today()).days


def _earnings_too_close(ticker: str) -> bool:
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return False
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0]
            elif not cal.empty:
                val = cal.iloc[0, 0]
            else:
                return False
        elif isinstance(cal, dict):
            val = cal.get("Earnings Date") or cal.get("earningsDate")
            if isinstance(val, list):
                val = val[0] if val else None
        else:
            return False
        if val is None:
            return False
        if hasattr(val, "date"):
            val = val.date()
        return abs((val - date.today()).days) <= _MAX_EARN_DAYS
    except Exception:
        return False


def _iv_too_expensive(ticker: str, iv: float) -> bool:
    try:
        hist = yf.Ticker(ticker).history(period="30d")
        if hist.empty or len(hist) < 15:
            return False
        hv = float(hist["Close"].pct_change().dropna().std() * math.sqrt(252))
        if hv <= 0:
            return False
        if iv / hv > _IV_HV_MAX_RATIO:
            logger.info(f"[options] {ticker} IV={iv:.2f} > {_IV_HV_MAX_RATIO}×HV={hv:.2f} — overpriced, skip")
            return True
        return False
    except Exception:
        return False


def _is_opening_flow(volume: int, oi: int) -> bool:
    if oi < _MIN_OI:
        return False
    if volume > oi:
        logger.info(f"[options] vol ({volume}) > OI ({oi}) — closing flow, skip")
        return False
    return True


def _unusual_volume_ratio(volume: int, oi: int) -> float:
    return min(volume / oi, 1.0) if oi > 0 else 0.0


# ---------------------------------------------------------------------------
# Polygon options chain (primary)
# ---------------------------------------------------------------------------

def _polygon_options_chain(ticker: str, direction: str, current_price: float) -> Optional[dict]:
    """
    Fetch options snapshot from Polygon. Returns a dict with keys matching
    the output format expected by scan(), or None if unavailable.
    """
    if not POLYGON_KEY:
        return None

    is_call = direction == "LONG"
    min_exp = (date.today() + timedelta(days=_MIN_DTE)).strftime("%Y-%m-%d")
    max_exp = (date.today() + timedelta(days=_MAX_DTE)).strftime("%Y-%m-%d")
    target_strike = _target_strike(current_price, is_call)

    try:
        r = _requests.get(
            f"https://api.polygon.io/v3/snapshot/options/{ticker}",
            params={
                "contract_type":        "call" if is_call else "put",
                "expiration_date.gte":  min_exp,
                "expiration_date.lte":  max_exp,
                "limit":                100,
                "apiKey":               POLYGON_KEY,
            },
            timeout=10,
        )
        if r.status_code != 200:
            logger.debug(f"[polygon options] {ticker}: HTTP {r.status_code}")
            return None

        items = r.json().get("results", [])
        if not items:
            return None

        candidates = []
        for item in items:
            details = item.get("details") or {}
            greeks  = item.get("greeks") or {}
            day     = item.get("day") or {}

            expiry = details.get("expiration_date", "")
            if not expiry:
                continue
            dte_val = _dte(expiry)
            if not (_MIN_DTE <= dte_val <= _MAX_DTE):
                continue

            strike = float(details.get("strike_price") or 0)
            oi     = int(item.get("open_interest") or 0)
            vol    = int(day.get("volume") or 0)
            prem   = float(day.get("last_price") or day.get("close") or 0)
            iv_raw = float(item.get("implied_volatility") or 0.30)

            if oi < _MIN_OI or prem < _MIN_ASK:
                continue

            delta = float(greeks.get("delta") or 0)
            theta = float(greeks.get("theta") or 0)

            # If Polygon didn't return Greeks, compute via Black-Scholes
            if delta == 0:
                T     = dte_val / 365.0
                delta = _bs_delta(current_price, strike, T, _RISK_FREE, iv_raw, is_call)
                theta = _bs_theta(current_price, strike, T, _RISK_FREE, iv_raw, is_call)

            candidates.append({
                "strike":        strike,
                "expiry":        expiry,
                "dte":           dte_val,
                "entry_premium": round(prem, 2),
                "delta":         round(delta, 3),
                "theta":         round(theta, 4),
                "iv":            round(iv_raw * 100, 1),
                "iv_raw":        iv_raw,
                "oi":            oi,
                "volume":        vol,
                "dist":          abs(strike - target_strike),
            })

        if not candidates:
            return None

        candidates.sort(key=lambda x: x["dist"])
        best = candidates[0]
        logger.info(
            f"[polygon options] {ticker} {'CALL' if is_call else 'PUT'} "
            f"strike={best['strike']} dte={best['dte']} prem={best['entry_premium']} "
            f"IV={best['iv']}% OI={best['oi']} vol={best['volume']}"
        )
        return best

    except Exception as e:
        logger.debug(f"[polygon options] {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# yfinance options chain (fallback)
# ---------------------------------------------------------------------------

def _yf_options_chain(ticker: str, direction: str, current_price: float) -> Optional[dict]:
    is_call = direction == "LONG"
    try:
        yf_ticker   = yf.Ticker(ticker)
        expirations = yf_ticker.options
        if not expirations:
            return None

        # Pick expiry in window
        target_expiry, dte_val = None, 0
        for exp in expirations:
            d = _dte(exp)
            if _MIN_DTE <= d <= _MAX_DTE:
                target_expiry, dte_val = exp, d
                break
        if not target_expiry:
            for exp in expirations:     # fallback window
                d = _dte(exp)
                if 14 <= d <= 90:
                    target_expiry, dte_val = exp, d
                    break
        if not target_expiry:
            return None

        chain     = yf_ticker.option_chain(target_expiry)
        contracts = (chain.calls if is_call else chain.puts).copy()
        if contracts.empty:
            return None

        target_strike = _target_strike(current_price, is_call)
        contracts["_dist"] = (contracts["strike"] - target_strike).abs()
        contracts = contracts[
            (contracts["ask"] >= _MIN_ASK) &
            (contracts["openInterest"] >= _MIN_OI)
        ].sort_values("_dist")

        if contracts.empty:
            return None

        row    = contracts.iloc[0]
        strike = float(row["strike"])
        prem   = round(float(row["ask"]), 2)
        iv_raw = float(row["impliedVolatility"]) if not pd.isna(row.get("impliedVolatility", float("nan"))) else 0.30
        oi     = int(row["openInterest"])
        vol    = int(row["volume"]) if not pd.isna(row.get("volume", float("nan"))) else 0

        T     = dte_val / 365.0
        delta = _bs_delta(current_price, strike, T, _RISK_FREE, iv_raw, is_call)
        theta = _bs_theta(current_price, strike, T, _RISK_FREE, iv_raw, is_call)

        return {
            "strike":        strike,
            "expiry":        target_expiry,
            "dte":           dte_val,
            "entry_premium": prem,
            "delta":         delta,
            "theta":         theta,
            "iv":            round(iv_raw * 100, 1),
            "iv_raw":        iv_raw,
            "oi":            oi,
            "volume":        vol,
            "dist":          abs(strike - target_strike),
        }

    except Exception as e:
        logger.debug(f"[yf options] {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def scan(ticker: str, direction: str, current_price: float,
         stock_target_one: Optional[float] = None) -> Optional[dict]:
    """
    Return option signal fields or None if any filter fails.
    Polygon is tried first; yfinance is the fallback.
    """
    try:
        # ── Filter 1: earnings proximity ──
        if _earnings_too_close(ticker):
            logger.info(f"[options] {ticker}: earnings too close — skip")
            return None

        # ── Fetch chain (Polygon primary → yfinance fallback) ──
        chain_data = _polygon_options_chain(ticker, direction, current_price)
        if not chain_data:
            logger.debug(f"[options] {ticker}: Polygon returned nothing — trying yfinance")
            chain_data = _yf_options_chain(ticker, direction, current_price)
        if not chain_data:
            logger.info(f"[options] {ticker}: no contract found in either source")
            return None

        strike        = chain_data["strike"]
        entry_premium = chain_data["entry_premium"]
        iv_raw        = chain_data["iv_raw"]
        oi            = chain_data["oi"]
        vol           = chain_data["volume"]
        dte_val       = chain_data["dte"]
        target_expiry = chain_data["expiry"]
        delta         = chain_data["delta"]
        theta         = chain_data["theta"]
        is_call       = direction == "LONG"

        # ── Filter 2: liquidity (already applied in chain fetch, double-check) ──
        if oi < _MIN_OI or entry_premium < _MIN_ASK:
            return None

        # ── Filter 3: flow validation ──
        if not _is_opening_flow(vol, oi):
            return None

        # ── Filter 4: IV vs HV ──
        if _iv_too_expensive(ticker, iv_raw):
            return None

        # ── Targets ──
        if stock_target_one and abs(delta) > 0.1:
            stock_move     = abs(stock_target_one - current_price)
            delta_gain     = abs(delta) * stock_move
            target_premium = round(max(entry_premium * 1.25, entry_premium + delta_gain), 2)
        else:
            target_premium = round(entry_premium * 1.35, 2)
        stop_premium = round(entry_premium * 0.75, 2)

        breakeven = round(strike + entry_premium if is_call else strike - entry_premium, 2)
        max_loss  = round(entry_premium * 100, 2)
        max_gain  = round((target_premium - entry_premium) * 100, 2)

        uv_ratio = _unusual_volume_ratio(vol, oi)
        logger.info(
            f"[options] {ticker} {'CALL' if is_call else 'PUT'} strike={strike} "
            f"dte={dte_val} prem={entry_premium} IV={chain_data['iv']}% "
            f"OI={oi} vol={vol} uv={uv_ratio:.2f} "
            f"target={target_premium} stop={stop_premium}"
        )

        return {
            "ticker":           ticker,
            "direction":        direction,
            "contract_type":    "CALL" if is_call else "PUT",
            "strike_price":     strike,
            "expiry_date":      target_expiry,
            "dte":              dte_val,
            "underlying_price": round(current_price, 2),
            "entry_premium":    entry_premium,
            "target_premium":   target_premium,
            "stop_premium":     stop_premium,
            "delta":            delta,
            "theta":            theta,
            "iv":               chain_data["iv"],
            "open_interest":    oi,
            "volume":           vol,
            "breakeven":        breakeven,
            "max_loss":         max_loss,
            "max_gain":         max_gain,
        }

    except Exception as e:
        logger.warning(f"[options] {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# LEAPS — long-dated, deep-ITM companion for the crash/deep-value signal (#10)
# ---------------------------------------------------------------------------
# Why deep-ITM and NOT OTM: deep-value fires IN a drawdown, when implied vol is
# spiked. An OTM call overpays for vega → vol-crush on recovery can lose money
# even as the stock rises. A deep-ITM LEAP (delta ~0.80) is mostly INTRINSIC
# value → behaves like leveraged stock with capped downside (premium paid), and
# barely cares about IV mean-reversion. Long DTE (1-2yr) gives the multi-month
# recovery thesis room to play out without theta pressure.
_LEAP_MIN_DTE        = 365     # >= 1 year
_LEAP_MAX_DTE        = 730     # <= 2 years
_LEAP_TARGET_DELTA   = 0.80    # deep-ITM target
_LEAP_DELTA_LO       = 0.65    # acceptable deep-ITM band
_LEAP_DELTA_HI       = 0.95
_LEAP_MIN_OI         = 100     # LEAPS are thinner than front-month — lower floor
_LEAP_MAX_SPREAD_PCT = 0.12    # reject illiquid wide markets: (ask-bid)/mid


def _spread_pct(bid: float, ask: float) -> Optional[float]:
    """(ask - bid) / mid, or None if a side is missing/zero (don't reject blind)."""
    if not bid or not ask or bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    return round((ask - bid) / mid, 4) if mid > 0 else None


def _pick_leap(candidates: list, target_delta: float = _LEAP_TARGET_DELTA) -> Optional[dict]:
    """From LEAP candidates pick the one whose |delta| is closest to the deep-ITM
    target, after enforcing the delta band + liquidity (OI, spread). PURE →
    unit-testable. Returns None when nothing is liquid + deep enough."""
    elig = [
        c for c in candidates
        if _LEAP_DELTA_LO <= abs(c.get("delta") or 0) <= _LEAP_DELTA_HI
        and (c.get("oi") or 0) >= _LEAP_MIN_OI
        and (c.get("spread_pct") is None or c["spread_pct"] <= _LEAP_MAX_SPREAD_PCT)
    ]
    if not elig:
        return None
    elig.sort(key=lambda c: abs(abs(c["delta"]) - target_delta))
    return elig[0]


def _polygon_leap_candidates(ticker: str, current_price: float) -> list:
    """Fetch deep-ITM long-dated CALLs from Polygon → list of candidate dicts."""
    if not POLYGON_KEY:
        return []
    min_exp = (date.today() + timedelta(days=_LEAP_MIN_DTE)).strftime("%Y-%m-%d")
    max_exp = (date.today() + timedelta(days=_LEAP_MAX_DTE)).strftime("%Y-%m-%d")
    try:
        r = _requests.get(
            f"https://api.polygon.io/v3/snapshot/options/{ticker}",
            params={
                "contract_type":       "call",
                "expiration_date.gte": min_exp,
                "expiration_date.lte": max_exp,
                "strike_price.lte":    round(current_price, 2),   # ITM calls only
                "limit":               250,
                "apiKey":              POLYGON_KEY,
            },
            timeout=12,
        )
        if r.status_code != 200:
            logger.debug(f"[leap polygon] {ticker}: HTTP {r.status_code}")
            return []
        out = []
        for item in r.json().get("results", []):
            details = item.get("details") or {}
            greeks  = item.get("greeks") or {}
            day     = item.get("day") or {}
            quote   = item.get("last_quote") or {}
            expiry  = details.get("expiration_date", "")
            if not expiry:
                continue
            dte_val = _dte(expiry)
            if not (_LEAP_MIN_DTE <= dte_val <= _LEAP_MAX_DTE):
                continue
            strike = float(details.get("strike_price") or 0)
            oi     = int(item.get("open_interest") or 0)
            vol    = int(day.get("volume") or 0)
            prem   = float(day.get("last_price") or day.get("close") or 0)
            iv_raw = float(item.get("implied_volatility") or 0.30)
            delta  = float(greeks.get("delta") or 0)
            theta  = float(greeks.get("theta") or 0)
            if delta == 0 and strike > 0:
                T = dte_val / 365.0
                delta = _bs_delta(current_price, strike, T, _RISK_FREE, iv_raw, True)
                theta = _bs_theta(current_price, strike, T, _RISK_FREE, iv_raw, True)
            bid = float(quote.get("bid") or 0)
            ask = float(quote.get("ask") or 0)
            # Fall back to last_price when no quote is present.
            if prem <= 0 and ask > 0:
                prem = round((bid + ask) / 2.0, 2)
            out.append({
                "strike": strike, "expiry": expiry, "dte": dte_val,
                "entry_premium": round(prem, 2), "delta": round(delta, 3),
                "theta": round(theta, 4), "iv": round(iv_raw * 100, 1), "iv_raw": iv_raw,
                "oi": oi, "volume": vol, "spread_pct": _spread_pct(bid, ask),
            })
        return out
    except Exception as e:
        logger.debug(f"[leap polygon] {ticker}: {e}")
        return []


def _yf_leap_candidates(ticker: str, current_price: float) -> list:
    """yfinance fallback: deep-ITM long-dated CALL candidates."""
    try:
        yt = yf.Ticker(ticker)
        exps = yt.options or []
        out = []
        for exp in exps:
            d = _dte(exp)
            if not (_LEAP_MIN_DTE <= d <= _LEAP_MAX_DTE):
                continue
            try:
                calls = yt.option_chain(exp).calls
            except Exception:
                continue
            itm = calls[calls["strike"] <= current_price]
            for _, row in itm.iterrows():
                strike = float(row["strike"])
                if strike <= 0:
                    continue
                iv_raw = float(row["impliedVolatility"]) if not pd.isna(row.get("impliedVolatility", float("nan"))) else 0.30
                oi     = int(row["openInterest"]) if not pd.isna(row.get("openInterest", float("nan"))) else 0
                vol    = int(row["volume"]) if not pd.isna(row.get("volume", float("nan"))) else 0
                bid    = float(row["bid"]) if not pd.isna(row.get("bid", float("nan"))) else 0.0
                ask    = float(row["ask"]) if not pd.isna(row.get("ask", float("nan"))) else 0.0
                prem   = round(ask, 2) if ask > 0 else (round(float(row["lastPrice"]), 2) if not pd.isna(row.get("lastPrice", float("nan"))) else 0.0)
                T      = d / 365.0
                delta  = _bs_delta(current_price, strike, T, _RISK_FREE, iv_raw, True)
                theta  = _bs_theta(current_price, strike, T, _RISK_FREE, iv_raw, True)
                out.append({
                    "strike": strike, "expiry": exp, "dte": d,
                    "entry_premium": prem, "delta": round(delta, 3), "theta": round(theta, 4),
                    "iv": round(iv_raw * 100, 1), "iv_raw": iv_raw, "oi": oi, "volume": vol,
                    "spread_pct": _spread_pct(bid, ask),
                })
        return out
    except Exception as e:
        logger.debug(f"[leap yf] {ticker}: {e}")
        return []


def scan_leap(ticker: str, current_price: float,
              stock_target: Optional[float] = None) -> Optional[dict]:
    """
    Find a deep-ITM LEAP CALL (1-2yr, delta ~0.80) for the crash/deep-value
    thesis. Returns option_signals row fields (same shape as scan()) or None.

    Deliberately does NOT apply the short-trade IV-vs-HV gate: a deep-ITM LEAP is
    intrinsic-dominated, so elevated crash IV barely affects it (that's the whole
    point). Liquidity (OI + bid/ask spread) IS gated — LEAPS get illiquid fast.
    """
    try:
        cands = _polygon_leap_candidates(ticker, current_price)
        if not cands:
            cands = _yf_leap_candidates(ticker, current_price)
        best = _pick_leap(cands)
        if not best:
            logger.info(f"[leap] {ticker}: no liquid deep-ITM LEAP found")
            return None

        strike = best["strike"]
        entry  = best["entry_premium"]
        delta  = best["delta"]
        dte_v  = best["dte"]
        if entry <= 0:
            return None

        # Recovery target: premium if the stock retraces toward `stock_target`
        # (deep-value t2 ≈ 95% of the prior 52-wk high). delta-scaled, intrinsic.
        if stock_target and abs(delta) > 0.1 and stock_target > current_price:
            target_prem = round(entry + abs(delta) * (stock_target - current_price), 2)
        else:
            target_prem = round(entry * 1.60, 2)   # generic multi-month recovery
        # Wide "disaster" reference only (manual mode — engine won't enforce it).
        stop_prem = round(entry * 0.50, 2)
        breakeven = round(strike + entry, 2)

        logger.info(
            f"[leap] {ticker} CALL strike={strike} dte={dte_v} Δ={delta} "
            f"prem={entry} IV={best['iv']}% OI={best['oi']} spread={best['spread_pct']}"
        )
        return {
            "ticker":           ticker,
            "direction":        "LONG",
            "contract_type":    "CALL",
            "strike_price":     strike,
            "expiry_date":      best["expiry"],
            "dte":              dte_v,
            "underlying_price": round(current_price, 2),
            "entry_premium":    entry,
            "target_premium":   target_prem,
            "stop_premium":     stop_prem,
            "delta":            delta,
            "theta":            best["theta"],
            "iv":               best["iv"],
            "open_interest":    best["oi"],
            "volume":           best["volume"],
            "breakeven":        breakeven,
            "max_loss":         round(entry * 100, 2),
            "max_gain":         round((target_prem - entry) * 100, 2),
        }
    except Exception as e:
        logger.warning(f"[leap] {ticker}: {e}")
        return None
