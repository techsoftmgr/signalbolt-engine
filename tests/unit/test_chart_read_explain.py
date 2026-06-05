"""Unit tests — chart_read._pattern_explain gives each detected pattern a
beginner-friendly plain-English read (so 'Bull Flag' isn't meaningless)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from engine.chart_read import _pattern_explain


def test_every_pattern_type_has_readable_explain():
    cases = [
        {"type": "Bull Flag", "target": 245.3},
        {"type": "Bear Flag", "target": 90.0},
        {"type": "Double Top", "level": 200.0, "neckline": 188.0, "target": 176.0},
        {"type": "Double Bottom", "level": 150.0, "neckline": 162.0, "target": 174.0},
        {"type": "Ascending Triangle", "upper": 200.0, "lower": 190.0, "target": 210.0},
        {"type": "Descending Triangle", "upper": 210.0, "lower": 200.0, "target": 190.0},
        {"type": "Symmetrical Triangle", "upper": 210.0, "lower": 190.0, "target": 220.0},
    ]
    for p in cases:
        txt = _pattern_explain(p)
        assert isinstance(txt, str) and len(txt) > 20          # a real sentence
        assert ("$" in txt) or ("target" in txt) or ("move" in txt)


def test_unknown_type_falls_back():
    txt = _pattern_explain({"type": "Wyckoff Spring", "target": 5.0})
    assert "Wyckoff Spring" in txt and "target" in txt


def test_missing_target_is_graceful():
    txt = _pattern_explain({"type": "Bull Flag"})
    assert "measured move" in txt
