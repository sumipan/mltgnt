"""mltgnt.media.webchat.inbound (#4032): POST body -> MediaEvent."""

from __future__ import annotations

import pytest

from mltgnt.media.webchat.inbound import DEFAULT_AUTHOR, to_media_event


def test_new_message_starts_its_own_thread() -> None:
    event = to_media_event({"text": " hello "}, space_id="webchat", message_id="m1")
    assert (event.space_id, event.conversation_id, event.message_id) == ("webchat", "webchat:m1", "m1")
    assert (event.author, event.text) == (DEFAULT_AUTHOR, "hello")
    assert event.attachments == ()


def test_reply_joins_the_thread() -> None:
    event = to_media_event({"text": "hi", "thread_ts": "root", "author": "u1"}, space_id="webchat", message_id="m2")
    assert event.conversation_id == "webchat:root"
    assert event.author == "u1"


def test_mention_is_left_for_the_host() -> None:
    body = {"text": "@helper please check"}
    event = to_media_event(body, space_id="webchat", message_id="m1")
    assert event.text == "@helper please check"
    assert event.raw == body
    assert event.raw is not body


@pytest.mark.parametrize(
    "body",
    [
        {"text": ""},
        {"text": "   "},
        {},
        {"text": 3},
        {"text": "x", "thread_ts": 5},
        {"text": "x", "author": ["a"]},
        ["text"],
        None,
    ],
)
def test_invalid_body_raises(body: object) -> None:
    with pytest.raises(ValueError):
        to_media_event(body, space_id="webchat", message_id="m1")


def test_blank_optional_fields_fall_back() -> None:
    event = to_media_event({"text": "x", "thread_ts": " ", "author": ""}, space_id="webchat", message_id="m1")
    assert event.conversation_id == "webchat:m1"
    assert event.author == DEFAULT_AUTHOR
