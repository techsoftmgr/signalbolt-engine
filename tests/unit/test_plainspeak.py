"""Unit tests — server-side terminology scrubber (plainspeak). Additive."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import plainspeak as ps


def test_phrases_rewritten():
    assert ps.plainify("MACD bullish crossover") == "Momentum turned up"
    assert "key pullback area" in ps.plainify("price held the golden pocket")
    assert ps.plainify("Reclaimed VWAP") == "Back above today's avg price"
    assert "Key levels" in ps.plainify("Fibonacci levels drawn")
    assert "no longer" not in ps.plainify("anything")  # sanity: no spurious text


def test_bare_tokens_word_boundary():
    # standalone tokens are rewritten
    assert "today's average price" in ps.plainify("crossed VWAP")
    assert ps.plainify("the Quant says") == "the Model says"
    # but NOT when they're substrings of a real word
    assert ps.plainify("Quantity surveyor") == "Quantity surveyor"
    assert ps.plainify("cinema") == "cinema"


def test_no_technical_words_leak_on_a_realistic_blob():
    txt = ("Technicals AGREE with the quant read. MACD crossed above its signal; "
           "price reclaimed VWAP and held the golden pocket. RSI overbought. "
           "The 9 EMA crossed above the 21 EMA. 1.618 extension target.")
    out = ps.plainify(txt)
    for jargon in ("MACD", "VWAP", "RSI", "golden pocket", "Fibonacci", "quant", "Quant"):
        assert jargon not in out, f"leaked: {jargon} in {out!r}"


def test_scrub_preserves_structure_numbers_and_symbols():
    obj = {
        "ticker": "ATR",                       # a symbol that collides with a token
        "price": 123.45,
        "bias": "down",
        "events": [
            {"title": "MACD bullish crossover", "price": 1.5, "tone": "bullish"},
            {"detail": "Lost VWAP at $10", "severity": 2},
        ],
        "note": "Golden pocket holding",
    }
    out = ps.scrub(obj)
    assert out["ticker"] == "ATR"              # symbol untouched
    assert out["price"] == 123.45              # number untouched
    assert out["bias"] == "down"
    assert out["events"][0]["title"] == "Momentum turned up"
    assert out["events"][0]["price"] == 1.5
    assert "VWAP" not in out["events"][1]["detail"]
    assert "Key pullback area" in out["note"] or "key pullback area" in out["note"]


def test_scrub_never_raises_on_weird_input():
    assert ps.scrub(None) is None
    assert ps.scrub(42) == 42
    assert ps.scrub({"a": {"b": ["MACD", 1, None]}})["a"]["b"][0] == "momentum"
