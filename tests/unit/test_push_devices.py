"""
Unit tests — engine/push.py multi-device token resolution.

The crux: one user with TWO devices must get push on BOTH (the bug was a single
profiles.push_token that the 2nd device overwrote). _merge_devices unions the
push_tokens table with the legacy column, deduped by token.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch
from engine import push

TOK_A = "ExponentPushToken[AAA]"
TOK_B = "ExponentPushToken[BBB]"
TOK_C = "ExponentPushToken[CCC]"
BAD   = "not-a-real-token"


class TestMergeDevices:
    def test_one_user_two_devices_both_included(self):
        ptoks = [{"user_id": "u1", "token": TOK_A}, {"user_id": "u1", "token": TOK_B}]
        profiles = [{"id": "u1", "push_token": TOK_A, "notification_prefs": {}, "email": "x@y.com"}]
        out = push._merge_devices(ptoks, profiles)
        tokens = {d["token"] for d in out}
        assert tokens == {TOK_A, TOK_B}          # both devices, deduped (A not doubled)
        assert all(d["user_id"] == "u1" for d in out)

    def test_legacy_token_included_when_not_in_table(self):
        # User has no push_tokens rows yet → fall back to profiles.push_token.
        out = push._merge_devices([], [{"id": "u2", "push_token": TOK_C, "email": "z@z.com"}])
        assert [d["token"] for d in out] == [TOK_C]

    def test_invalid_tokens_dropped(self):
        out = push._merge_devices([{"user_id": "u", "token": BAD}],
                                  [{"id": "u", "push_token": None}])
        assert out == []

    def test_prefs_merged_with_defaults_and_email_attached(self):
        ptoks = [{"user_id": "u1", "token": TOK_A}]
        profiles = [{"id": "u1", "push_token": None,
                     "notification_prefs": {"watchlist_alerts": False}, "email": "a@b.com"}]
        out = push._merge_devices(ptoks, profiles)
        d = out[0]
        assert d["email"] == "a@b.com"
        assert d["prefs"]["watchlist_alerts"] is False          # override honored
        assert d["prefs"].get("new_signals") is True            # default filled in

    def test_dedup_token_across_sources(self):
        ptoks = [{"user_id": "u1", "token": TOK_A}]
        profiles = [{"id": "u1", "push_token": TOK_A, "email": "a@b.com"}]
        out = push._merge_devices(ptoks, profiles)
        assert len(out) == 1                                     # same token once


_DEVICES = [
    {"user_id": "u1", "token": TOK_A, "prefs": {**push._DEFAULT_PREFS, "watchlist_alerts": True},  "email": "u1@x.com"},
    {"user_id": "u1", "token": TOK_B, "prefs": {**push._DEFAULT_PREFS, "watchlist_alerts": True},  "email": "u1@x.com"},
    {"user_id": "u2", "token": TOK_C, "prefs": {**push._DEFAULT_PREFS, "watchlist_alerts": False}, "email": "admin@x.com"},
]


class TestTokenSelectors:
    def test_tokens_for_users_returns_all_devices(self):
        with patch("engine.push._device_rows", return_value=_DEVICES):
            toks = push._tokens_for_users(["u1"])
            assert set(toks) == {TOK_A, TOK_B}          # both of u1's devices

    def test_tokens_for_users_pref_gated(self):
        with patch("engine.push._device_rows", return_value=_DEVICES):
            toks = push._tokens_for_users(["u1", "u2"], "watchlist_alerts")
            assert set(toks) == {TOK_A, TOK_B}          # u2 has the pref OFF

    def test_tokens_for_empty_users(self):
        with patch("engine.push._device_rows", return_value=_DEVICES):
            assert push._tokens_for_users([]) == []

    def test_get_profiles_shim_shape(self):
        with patch("engine.push._device_rows", return_value=_DEVICES):
            profs = push._get_profiles()
            assert len(profs) == 3 and set(profs[0]) == {"token", "prefs"}
