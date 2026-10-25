"""HubClient and guard / filter behaviour."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pytest

from mltgnt.interfaces.hub import HubAddress, HubMessage, HubTransport
from mltgnt.interfaces.media import MediaClient, Status
from mltgnt.media.hub.client import HubClient, UrllibTransport
from mltgnt.media.hub.config import HubMediaConfig
from mltgnt.media.hub.guard import GuardConfig


def _config(tmp_path: Path, **kwargs: object) -> HubMediaConfig:
    defaults = {
        "state_dir": tmp_path / "s",
        "pending_dir": tmp_path / "p",
        "events_dir": tmp_path / "e",
        "hub_url": "http://127.0.0.1:8790",
        "host_id": "host-a",
        "token_env": "HUB_TOKEN",
        "rooms": ("r1",),
        "guard": GuardConfig(hop_limit=3, max_agent_run=4, min_interval_sec=5.0, room_daily_limit=200),
    }
    defaults.update(kwargs)
    return HubMediaConfig(**defaults)  # type: ignore[arg-type]


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[HubMessage] = []
        self.guard_calls: list[tuple[str, str | None]] = []
        self.thread_rows: list[Mapping[str, Any]] = []
        self.room_today_count = 0
        self.fail_send = False
        self.fail_guard = False
        self.revise_calls: list[tuple[str, str, str, dict[str, Any]]] = []
        self.raise_on_send = False
        self.raise_on_revise = False
        self.raise_on_guard = False

    def send(self, message: HubMessage) -> str | None:
        if self.raise_on_send:
            raise RuntimeError("send boom")
        if self.fail_send:
            return None
        self.sent.append(message)
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
        if self.raise_on_revise:
            raise RuntimeError("revise boom")
        self.revise_calls.append((room, message_id, kind, {"text": text, "status": status}))
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
        self.guard_calls.append((room, thread))
        if self.raise_on_guard:
            raise RuntimeError("guard boom")
        if self.fail_guard:
            return None
        return self.thread_rows, self.room_today_count


def test_satisfies_media_client(tmp_path: Path) -> None:
    transport = FakeTransport()
    client = HubClient(_config(tmp_path), persona="persona-x", transport=transport)
    assert isinstance(client, MediaClient)


def test_post_envelope_and_guard_context(tmp_path: Path) -> None:
    transport = FakeTransport()
    client = HubClient(_config(tmp_path), persona="persona-x", transport=transport)
    message_id = client.post(
        "hi",
        "hub.r1",
        None,
        extra={"to": {"host": "host-b", "persona": "persona-y"}, "hop": 1},
    )
    assert message_id is not None
    uuid.UUID(message_id)
    assert transport.guard_calls == [("r1", None)]
    assert len(transport.sent) == 1
    msg = transport.sent[0]
    assert msg.room == "r1"
    assert msg.to == HubAddress("host-b", "persona-y")
    assert msg.hop == 1
    assert msg.ts is None


def test_post_reuses_pending_id_on_send_retry(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.fail_send = True
    client = HubClient(_config(tmp_path), persona=None, transport=transport)
    assert client.post("hi", "hub.r1") is None
    transport.fail_send = False
    message_id = client.post("hi", "hub.r1")
    assert message_id is not None
    assert len(transport.sent) == 1
    assert transport.sent[0].id == message_id


def test_outbound_filter_blocks_send(tmp_path: Path) -> None:
    transport = FakeTransport()

    def block(_text: str) -> str | None:
        return None

    config = _config(tmp_path, outbound_filter=block)
    client = HubClient(config, persona=None, transport=transport)
    assert client.post("hi", "hub.r1") is None
    assert transport.sent == []
    assert transport.guard_calls == []


def test_guard_rejection_skips_send(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.room_today_count = 9999
    client = HubClient(_config(tmp_path), persona="bot", transport=transport)
    assert client.post("hi", "hub.r1") is None
    assert transport.sent == []


def test_guard_context_failure_skips_send(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.fail_guard = True
    client = HubClient(_config(tmp_path), persona=None, transport=transport)
    assert client.post("hi", "hub.r1") is None
    assert transport.sent == []


def test_update_and_set_status_known_id(tmp_path: Path) -> None:
    transport = FakeTransport()
    client = HubClient(_config(tmp_path), persona=None, transport=transport)
    message_id = client.post("v1", "hub.r1")
    assert message_id is not None
    assert client.update(message_id, "v2") is True
    assert client.set_status(message_id, Status.DONE) is True
    assert transport.revise_calls[0][:3] == ("r1", message_id, "update")
    assert transport.revise_calls[1][:3] == ("r1", message_id, "status")


def test_unknown_id_revise_is_false(tmp_path: Path) -> None:
    transport = FakeTransport()
    client = HubClient(_config(tmp_path), persona=None, transport=transport)
    assert client.update("missing", "x") is False
    assert client.set_status("missing", Status.WORKING) is False


def test_transport_exceptions_are_swallowed(tmp_path: Path) -> None:
    transport = FakeTransport()
    client = HubClient(_config(tmp_path), persona=None, transport=transport)
    message_id = client.post("v1", "hub.r1")
    assert message_id is not None
    transport.raise_on_send = True
    assert client.post("v2", "hub.r1") is None
    transport.raise_on_revise = True
    assert client.update(message_id, "v3") is False
    assert client.set_status(message_id, Status.FAILED) is False


def test_guard_and_filter_exceptions_are_swallowed(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.raise_on_guard = True
    client = HubClient(_config(tmp_path), persona=None, transport=transport)
    assert client.post("v1", "hub.r1") is None
    assert transport.sent == []

    def broken_filter(_text: str) -> str | None:
        raise RuntimeError("filter boom")

    filtered_client = HubClient(
        _config(tmp_path, outbound_filter=broken_filter),
        persona=None,
        transport=FakeTransport(),
    )
    assert filtered_client.post("v2", "hub.r1") is None


@pytest.mark.parametrize("timeout_sec", [0, -1, float("inf"), float("nan")])
def test_transport_rejects_non_finite_or_non_positive_timeout(tmp_path: Path, timeout_sec: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        UrllibTransport(_config(tmp_path), timeout_sec=timeout_sec)


def test_upload_and_react_unsupported(tmp_path: Path) -> None:
    client = HubClient(_config(tmp_path), persona=None, transport=FakeTransport())
    assert client.upload("x", "hub.r1") is False
    assert client.react("id", "thumbsup") is False


def test_post_carries_display(tmp_path: Path) -> None:
    from mltgnt.interfaces.hub import HubDisplay
    from mltgnt.media.hub.client import _message_to_post_body

    display = HubDisplay(name="Staff A", icon_url="https://example.com/a.png")
    transport = FakeTransport()
    client = HubClient(_config(tmp_path), persona="persona-x", transport=transport, display=display)
    assert client.post("hi", "hub.r1") is not None
    msg = transport.sent[0]
    assert msg.display == display
    assert _message_to_post_body(msg)["display"] == {"name": "Staff A", "icon_url": "https://example.com/a.png"}

    bare = FakeTransport()
    HubClient(_config(tmp_path), persona="persona-x", transport=bare).post("hi", "hub.r1")
    assert bare.sent[0].display is None
    assert "display" not in _message_to_post_body(bare.sent[0])
