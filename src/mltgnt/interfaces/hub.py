"""Hub envelope types and transport boundary (no media-layer imports)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from mltgnt.interfaces.media import Status
from mltgnt.interfaces.turn import Attachment

__all__ = ["HubAddress", "HubMessage", "HubTransport"]

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_HUB_KINDS = frozenset({"message", "status", "update"})
_ROW_CORE_KEYS = frozenset(
    {
        "ts",
        "message_id",
        "author",
        "text",
        "thread_ts",
        "kind",
        "status",
        "task_uuid",
        "bookmarked",
        "reaction",
        "files",
    }
)


def _validate_identifier(value: str, *, field: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"invalid {field}: {value!r}")
    if ".." in value:
        raise ValueError(f"invalid {field}: {value!r}")
    return value


def _validate_persona(value: str) -> str:
    if not value or len(value) > 64:
        raise ValueError(f"invalid persona: {value!r}")
    if "/" in value or ":" in value or any(ord(c) < 32 for c in value):
        raise ValueError(f"invalid persona: {value!r}")
    return value


def _address_to_author(host: str, persona: str | None) -> str:
    _validate_identifier(host, field="host")
    if persona is not None:
        _validate_persona(persona)
        return f"{host}/{persona}"
    return f"{host}/user"


def _author_to_address(author: str) -> HubAddress:
    if "/" not in author:
        raise ValueError(f"invalid author: {author!r}")
    host, tail = author.split("/", 1)
    _validate_identifier(host, field="host")
    if tail == "user":
        return HubAddress(host, None)
    _validate_persona(tail)
    return HubAddress(host, tail)


def _address_to_mapping(addr: HubAddress) -> dict[str, Any]:
    out: dict[str, Any] = {"host": addr.host}
    if addr.persona is not None:
        out["persona"] = addr.persona
    return out


def _mapping_to_address(data: Any) -> HubAddress | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"invalid address mapping: {data!r}")
    host = data.get("host")
    if not isinstance(host, str):
        raise ValueError(f"invalid address mapping: {data!r}")
    persona = data.get("persona")
    if persona is not None and not isinstance(persona, str):
        raise ValueError(f"invalid address mapping: {data!r}")
    return HubAddress(host, persona)


@dataclass(frozen=True)
class HubAddress:
    host: str
    persona: str | None

    def __post_init__(self) -> None:
        _validate_identifier(self.host, field="host")
        if self.persona is not None:
            _validate_persona(self.persona)


@dataclass(frozen=True)
class HubMessage:
    id: str
    ts: str | None
    room: str
    thread: str | None
    sender: HubAddress
    to: HubAddress | None
    text: str
    attachments: tuple[Attachment, ...]
    hop: int
    origin_id: str | None
    kind: Literal["message", "status", "update"]
    status: str | None

    def to_row(self) -> dict[str, Any]:
        _validate_identifier(self.room, field="room")
        if self.hop < 0:
            raise ValueError(f"invalid hop: {self.hop}")
        if self.kind not in _HUB_KINDS:
            raise ValueError(f"unknown kind: {self.kind!r}")
        row: dict[str, Any] = {
            "message_id": self.id,
            "author": _address_to_author(self.sender.host, self.sender.persona),
            "text": self.text,
            "thread_ts": self.thread,
            "kind": self.kind,
            "status": self.status,
            "room": self.room,
            "sender": _address_to_mapping(self.sender),
            "hop": self.hop,
        }
        if self.ts is not None:
            row["ts"] = self.ts
        if self.to is not None:
            row["to"] = _address_to_mapping(self.to)
        if self.origin_id is not None:
            row["origin_id"] = self.origin_id
        if self.attachments:
            row["files"] = [
                {
                    k: v
                    for k, v in (
                        ("name", att.name),
                        ("content_type", att.content_type),
                        ("uri", att.uri),
                    )
                    if v is not None
                }
                for att in self.attachments
            ]
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> HubMessage:
        kind = row.get("kind")
        if kind not in _HUB_KINDS:
            raise ValueError(f"unknown kind: {kind!r}")
        room = row.get("room")
        if not isinstance(room, str):
            raise ValueError(f"invalid room: {room!r}")
        _validate_identifier(room, field="room")
        hop = row.get("hop", 0)
        if not isinstance(hop, int) or hop < 0:
            raise ValueError(f"invalid hop: {hop!r}")
        message_id = row.get("message_id")
        if not isinstance(message_id, str) or not message_id:
            raise ValueError(f"invalid message_id: {message_id!r}")
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"invalid text: {text!r}")
        thread = row.get("thread_ts")
        if thread is not None and not isinstance(thread, str):
            raise ValueError(f"invalid thread_ts: {thread!r}")
        ts = row.get("ts")
        if ts is not None and not isinstance(ts, str):
            raise ValueError(f"invalid ts: {ts!r}")
        status = row.get("status")
        if status is not None and not isinstance(status, str):
            raise ValueError(f"invalid status: {status!r}")
        sender_data = row.get("sender")
        if sender_data is not None:
            sender = _mapping_to_address(sender_data)
            if sender is None:
                raise ValueError("invalid sender")
        else:
            author = row.get("author")
            if not isinstance(author, str):
                raise ValueError(f"invalid author: {author!r}")
            sender = _author_to_address(author)
        to = _mapping_to_address(row.get("to"))
        origin_id = row.get("origin_id")
        if origin_id is not None and not isinstance(origin_id, str):
            raise ValueError(f"invalid origin_id: {origin_id!r}")
        attachments: tuple[Attachment, ...] = ()
        files = row.get("files")
        if files:
            if not isinstance(files, list):
                raise ValueError("invalid files")
            parsed: list[Attachment] = []
            for item in files:
                if not isinstance(item, dict):
                    raise ValueError("invalid files")
                name = item.get("name")
                if not isinstance(name, str):
                    raise ValueError("invalid files")
                parsed.append(
                    Attachment(
                        name=name,
                        content_type=item.get("content_type"),
                        uri=item.get("uri"),
                    )
                )
            attachments = tuple(parsed)
        return cls(
            id=message_id,
            ts=ts,
            room=room,
            thread=thread,
            sender=sender,
            to=to,
            text=text,
            attachments=attachments,
            hop=hop,
            origin_id=origin_id,
            kind=kind,
            status=status,
        )


@runtime_checkable
class HubTransport(Protocol):
    def send(self, message: HubMessage) -> str | None:
        ...

    def revise(
        self,
        room: str,
        message_id: str,
        *,
        kind: Literal["update", "status"],
        text: str | None = None,
        status: Status | None = None,
    ) -> bool:
        ...

    def poll(
        self,
        room: str,
        after: str | None,
        *,
        limit: int = 100,
    ) -> tuple[list[HubMessage], str | None] | None:
        ...

    def guard_context(
        self,
        room: str,
        thread: str | None,
    ) -> tuple[list[Mapping[str, Any]], int] | None:
        ...
