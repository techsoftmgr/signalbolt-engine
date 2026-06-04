"""
Unit tests — engine/heatmap_service._expected_volume_fraction (intraday volume
curve). Regression for the HOOD 2026-06-04 false "2.3x volume" accum signal at
9:46am: the old naive elapsed/390 (floored at 0.05) assumed ~5% of volume done
16 min in, when ~14% really is — over-projecting relative volume ~2.9x.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.heatmap_service import _expected_volume_fraction as ef


class TestVolumeCurve:
    def test_monotonic_increasing(self):
        prev = -1.0
        for m in range(5, 391, 5):
            v = ef(m)
            assert v >= prev, f"curve dipped at {m}min"
            prev = v

    def test_full_day_is_one(self):
        assert ef(390) == 1.0
        assert ef(500) == 1.0

    def test_floor_at_open(self):
        # First ~5 min floored (avoids div-by-zero + first-bar noise)
        assert ef(0) == ef(5) > 0.0
        assert abs(ef(5) - 0.087) < 1e-9

    def test_first_15_min_is_front_loaded(self):
        # ~14% of the day's volume by 15 min — NOT the ~4% the clock implies.
        assert 0.13 <= ef(15) <= 0.15

    def test_16min_matches_hood_case(self):
        # Interp between 15min(0.139) and 20min(0.160): ~0.143
        v = ef(16)
        assert 0.14 <= v <= 0.146

    def test_debiases_vs_old_linear(self):
        # The old code used max(16/390, 0.05) = 0.05. A signal the old math called
        # "2.3x" at 16min becomes 2.3 * (0.05 / ef(16)) with the curve.
        old_frac = 0.05
        rel_old = 2.3
        rel_new = rel_old * old_frac / ef(16)
        assert rel_new < 1.0          # HOOD's real opening volume was BELOW average
        assert rel_new < rel_old      # always de-biases downward early in the session

    def test_midday_reasonable(self):
        # ~half the day's volume by ~2 hours in.
        assert 0.40 <= ef(120) <= 0.50
