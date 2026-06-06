"""Unit tests — regime bucketing + the advisory detector_policy recommender."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine import regime_buckets as rb
from engine import detector_policy as dp


def test_bucket_mapping():
    assert rb.bucket_of("TRENDING_BULL") == rb.RISK_ON
    assert rb.bucket_of("LOW_VOL") == rb.RISK_ON
    assert rb.bucket_of("RANGING") == rb.NEUTRAL
    assert rb.bucket_of("PANIC") == rb.RISK_OFF
    assert rb.bucket_of("TRENDING_BEAR") == rb.RISK_OFF
    assert rb.bucket_of("HIGH_VOL") == rb.RISK_OFF
    assert rb.bucket_of("") == "ANY" and rb.bucket_of(None) == "ANY"
    assert rb.bucket_of("RISK_ON") == "RISK_ON"          # already a bucket → itself
    assert rb.bucket_of("nonsense") == "ANY"


def _row(det, regime, pct, alpha=None):
    sb = {"detector_source": det}
    if alpha is not None:
        sb["alpha_pct"] = alpha
    return {"result_pct": pct, "regime_type": regime, "score_breakdown": sb, "strategy_type": "swing"}


def test_policy_sample_floor_keeps_full_size():
    rows = [_row("NEWDET", "TRENDING_BULL", -5.0) for _ in range(5)]   # bad but n<floor
    pol = {p["detector"]: p for p in dp.recommend(rows)}
    assert pol["NEWDET"]["action"] == "MEASURING"
    assert pol["NEWDET"]["recommended_multiplier"] == 1.0     # never act on noise


def test_policy_throttles_confirmed_loser_not_kill():
    rows = [_row("LOSER", "PANIC", -2.0) for _ in range(45)]   # negative, n>=confirm
    p = dp.recommend(rows)[0]
    assert p["detector"] == "LOSER" and p["bucket"] == "RISK_OFF"
    assert p["action"] == "THROTTLE" and p["recommended_multiplier"] == 0.25   # shrink, not zero


def test_policy_full_size_for_real_alpha():
    rows = [_row("WINNER", "RISK_OFF", 2.0, alpha=1.5) for _ in range(30)]
    p = {x["detector"]: x for x in dp.recommend(rows)}["WINNER"]
    assert p["action"] == "FULL" and p["recommended_multiplier"] == 1.0
    assert "alpha" in p["note"]


def test_policy_flags_beta_only_winner():
    rows = [_row("BETA", "TRENDING_BULL", 2.0, alpha=-0.5) for _ in range(30)]
    p = {x["detector"]: x for x in dp.recommend(rows)}["BETA"]
    assert p["action"] == "FULL" and "BETA-only" in p["note"]


def test_policy_buckets_collapse_regimes():
    # PANIC + HIGH_VOL + TRENDING_BEAR all fold into one RISK_OFF cell
    rows = ([_row("D", "PANIC", -2.0) for _ in range(15)] +
            [_row("D", "HIGH_VOL", -2.0) for _ in range(15)] +
            [_row("D", "TRENDING_BEAR", -2.0) for _ in range(15)])
    pol = dp.recommend(rows)
    risk_off = [p for p in pol if p["bucket"] == "RISK_OFF"]
    assert len(risk_off) == 1 and risk_off[0]["n"] == 45      # 3 regimes → 1 bucket cell
