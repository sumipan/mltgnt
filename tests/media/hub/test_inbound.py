"""Hub inbound -> MediaEvent."""

from __future__ import annotations

from mltgnt.interfaces.hub import HubAddress, HubMessage
from mltgnt.media.hub.inbound import to_media_event


def _message(**kwargs: object) -> HubMessage:
    defaults = {
        "id": "m1",
        "ts": None,
        "room": "r1",
        "thread": None,
        "sender": HubAddress("host-a", "persona-x"),
        "to": None,
        "text": "hello",
        "attachments": (),
        "hop": 0,
        "origin_id": None,
        "kind": "message",
        "status": None,
    }
    defaults.update(kwargs)
    return HubMessage(**defaults)  # type: ignore[arg-type]


def test_to_media_event_with_persona() -> None:
    event = to_media_event(_message(), space_prefix="hub.")
    assert event.space_id == "hub.r1"
    assert event.conversation_id == "hub.r1:m1"
    assert event.author == "host-a/persona-x"
    assert event.text.startswith("[host-a/persona-x] ")


def test_to_media_event_without_persona() -> None:
    event = to_media_event(_message(sender=HubAddress("host-a", None)), space_prefix="hub.")
    assert event.author == "host-a/user"
    assert event.text.startswith("[host-a/user] ")
