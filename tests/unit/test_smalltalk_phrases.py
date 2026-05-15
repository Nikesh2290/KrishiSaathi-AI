"""Smalltalk fast-path phrase coverage."""

from __future__ import annotations

import pytest

from api.routes import query as query_module


@pytest.mark.parametrize(
    "text",
    [
        "thank you so much",
        "Thank you very much",
        "thanks a lot ji",
        "thanks",
        "bahut bahut dhanyawad",
        "hi Krishi ji",
        "hello ji",
        "namaste bhai",
        "good morning",
        "bye bhai",
        "see you later",
        "ok got it",
        "theek hai samajh gaya",
    ],
)
def test_smalltalk_positive(text):
    out = query_module._smalltalk_reply(text, farmer_name="Ram")
    assert out is not None, f"expected match for {text!r}"
    kind, msg = out
    assert kind.startswith("smalltalk_")
    assert msg.strip()


def test_smalltalk_thanks_with_farming_not_matched():
    assert query_module._smalltalk_reply("thanks, meri gehu mein bimari hai") is None


def test_smalltalk_farmer_name_in_thanks():
    out = query_module._smalltalk_reply("thank you so much", farmer_name="Ramesh")
    assert out is not None
    _kind, msg = out
    assert "Ramesh" in msg or "रमेश" in msg


def test_greeting_hi_plain():
    out = query_module._smalltalk_reply("hi", farmer_name=None)
    assert out is not None
    assert out[0] == "smalltalk_greeting"
