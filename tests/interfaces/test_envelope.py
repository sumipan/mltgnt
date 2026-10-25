"""Hub envelope round-trip and HubTransport protocol checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pytest

from mltgnt.interfaces.hub import HubAddress, HubMessage, HubTransport
from mltgnt.interfaces.media import Status
from mltgnt.interfaces.turn import Attachment


def _base_message(**overrides: object) -> HubMessage:
    defaults: dict[str, object] = {
        "id": "msg-1",
        "ts": "2026-01-05T10:00:00+00:00",
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
    defaults.update(overrides)
    return HubMessage(**defaults)  # type: ignore[arg-type]


def test_round_trip_to_none() -> None:
    m = _base_message(to=None)
    assert HubMessage.from_row(m.to_row()) == m


def test_round_trip_thread_none() -> None:
    m = _base_message(thread=None)
    assert HubMessage.from_row(m.to_row()) == m


def test_round_trip_with_attachments() -> None:
    att = Attachment(name="a.png", content_type="image/png", uri="file:///a.png")
    m = _base_message(attachments=(att,))
    assert HubMessage.from_row(m.to_row()) == m


def test_round_trip_kind_status() -> None:
    m = _base_message(kind="status", status=Status.WORKING.value)
    assert HubMessage.from_row(m.to_row()) == m


def test_round_trip_with_to_address() -> None:
    m = _base_message(to=HubAddress("host-b", "persona-y"), hop=1)
    assert HubMessage.from_row(m.to_row()) == m


class _FullTransport:
    def send(self, message: HubMessage) -> str | None:
        return message.id

    def revise(
        self,
        room: str,
        message_id: str,
        *,
        kind: Literal["update", "status"],
        text: str | None = None,
        status: Status | None = None,
    ) -> bool:
        return True

    def poll(
        self,
        room: str,
        after: str | None,
        *,
        limit: int = 100,
    ) -> tuple[list[HubMessage], str | None] | None:
        return [], None

    def guard_context(
        self,
        room: str,
        thread: str | None,
    ) -> tuple[list[Mapping[str, Any]], int] | None:
        return [], 0


def test_hub_transport_isinstance() -> None:
    assert isinstance(_FullTransport(), HubTransport)


@pytest.mark.parametrize("missing", ["send", "revise", "poll", "guard_context"])
def test_hub_transport_rejects_each_missing_method(missing: str) -> None:
    methods = {
        name: (lambda self, *args, **kwargs: None)
        for name in ("send", "revise", "poll", "guard_context")
        if name != missing
    }
    incomplete = type("IncompleteTransport", (), methods)()
    assert not isinstance(incomplete, HubTransport)


def test_hub_address_accepts_valid() -> None:
    HubAddress("host-a", "persona-x")


@pytest.mark.parametrize(
    "host",
    ["", "../r1", "a:b"],
)
def test_hub_address_rejects_invalid_host(host: str) -> None:
    with pytest.raises(ValueError):
        HubAddress(host, "persona-x")


@pytest.mark.parametrize(
    "persona",
    ["", "a/b", "a:b"],
)
def test_hub_address_rejects_invalid_persona(persona: str) -> None:
    with pytest.raises(ValueError):
        HubAddress("host-a", persona)


def test_from_row_rejects_bad_room() -> None:
    m = _base_message()
    row = m.to_row()
    row["room"] = "../escape"
    with pytest.raises(ValueError):
        HubMessage.from_row(row)


def test_from_row_rejects_negative_hop() -> None:
    m = _base_message()
    row = m.to_row()
    row["hop"] = -1
    with pytest.raises(ValueError):
        HubMessage.from_row(row)


def test_from_row_rejects_unknown_kind() -> None:
    m = _base_message()
    row = m.to_row()
    row["kind"] = "bookmark"
    with pytest.raises(ValueError):
        HubMessage.from_row(row)


def test_human_sender_author() -> None:
    m = _base_message(sender=HubAddress("host-a", None))
    row = m.to_row()
    assert row["author"] == "host-a/user"
    assert HubMessage.from_row(row).sender == HubAddress("host-a", None)


def test_display_round_trips_and_is_optional() -> None:
    from mltgnt.interfaces.hub import HubDisplay

    plain = _base_message()
    assert "display" not in plain.to_row()
    assert HubMessage.from_row(plain.to_row()).display is None

    shown = _base_message(display=HubDisplay(name="Staff A", icon_url="https://example.com/a.png"))
    row = shown.to_row()
    assert row["display"] == {"name": "Staff A", "icon_url": "https://example.com/a.png"}
    assert HubMessage.from_row(row) == shown

    name_only = _base_message(display=HubDisplay(name="Staff A"))
    assert name_only.to_row()["display"] == {"name": "Staff A"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": ""},
        {"name": "a\nb"},
        {"name": "x" * 81},
        {"icon_url": "ftp://example.com/a.png"},
        {"icon_url": "javascript:alert(1)"},
        {"icon_url": "https://example.com/a b.png"},
    ],
)
def test_display_rejects_invalid(kwargs: dict[str, str]) -> None:
    from mltgnt.interfaces.hub import HubDisplay

    with pytest.raises(ValueError):
        HubDisplay(**kwargs)


def test_display_from_mapping_rejects_unknown_keys() -> None:
    from mltgnt.interfaces.hub import HubDisplay

    with pytest.raises(ValueError):
        HubDisplay.from_mapping({"name": "a", "color": "red"})
    assert HubDisplay.from_mapping({}) is None
    assert HubDisplay.from_mapping(None) is None
