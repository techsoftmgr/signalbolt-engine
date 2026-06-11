"""Unit tests — community headline relevance filter. Offline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import social_insights as si


def test_headline_relevant_symbol_and_name():
    # symbol as a word
    assert si._headline_relevant("MU shares climb after trading signal", "MU", "Micron Technology")
    # company-name token
    assert si._headline_relevant("Micron sees strong demand", "MU", "Micron Technology Inc")
    # $TICKER form
    assert si._headline_relevant("Buying $ADBE on the dip", "ADBE", "Adobe Inc")


def test_headline_relevant_rejects_generic_roundup():
    # generic market roundup that merely tags MSFT/TSLA — no symbol/name match
    assert not si._headline_relevant("Nvidia Millionaires Can't Afford To Sell, ETFs May Help", "MSFT", "Microsoft Corp")
    assert not si._headline_relevant("Stock market whipsawed on Trump statements", "TSLA", "Tesla Inc")
    # 'mu' must match as a WORD, not inside another word (e.g. 'museum')
    assert not si._headline_relevant("New museum opens downtown", "MU", "Micron Technology")


def test_name_tokens_drops_suffixes():
    toks = [t.lower() for t in si._name_tokens("Adobe Inc")]
    assert "adobe" in toks and "inc" not in toks
