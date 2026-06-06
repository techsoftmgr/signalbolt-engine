"""Unit tests — social_feed (Discord/TweetShift reader) + tape wiring. Additive."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine import social_feed as sf
from engine import market_commentary as mc


def test_parse_tweetshift_embed():
    msg = {
        "id": "123",
        "timestamp": "2026-06-06T13:00:00Z",
        "embeds": [{
            "description": "Big tariffs coming on imports. Many people are saying!",
            "url": "https://x.com/realDonaldTrump/status/999",
            "timestamp": "2026-06-06T12:59:00Z",
            "author": {"name": "Donald J. Trump (@realDonaldTrump)"},
        }],
    }
    p = sf._parse_message(msg)
    assert p and "tariffs" in p["text"].lower()
    assert "realDonaldTrump" in p["author"]
    assert p["url"].endswith("/999")
    assert p["created_at"] == "2026-06-06T12:59:00Z"


def test_parse_plain_content_fallback():
    p = sf._parse_message({"id": "1", "content": "Fed to cut rates", "timestamp": "t", "embeds": []})
    assert p and p["text"] == "Fed to cut rates"


def test_parse_empty_returns_none():
    assert sf._parse_message({"id": "1", "embeds": [], "content": ""}) is None


def test_recent_posts_unconfigured(monkeypatch):
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SOCIAL_FEED_CHANNEL_ID", raising=False)
    assert sf.is_configured() is False
    assert sf.recent_posts() == []


def test_social_events_map_to_high_severity(monkeypatch):
    monkeypatch.setattr(sf, "recent_posts", lambda limit=6: [
        {"text": "New tariffs announced", "author": "@realDonaldTrump",
         "url": "https://x.com/x/status/1", "created_at": "2026-06-06T12:00:00Z"},
    ])
    out = mc._social_events(6)
    assert out and out[0]["type"] == "SOCIAL"
    assert out[0]["severity"] == 3
    assert "tariffs" in out[0]["detail"].lower()
    assert out[0]["url"].endswith("/1")


def test_social_events_empty_on_failure(monkeypatch):
    monkeypatch.setattr(sf, "recent_posts", lambda limit=6: (_ for _ in ()).throw(RuntimeError("x")))
    assert mc._social_events() == []
