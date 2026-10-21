"""Hub media client / subscriber settings (stdlib only; no FastAPI import)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mltgnt.media._core.config import MediaConfig
from mltgnt.media.hub.guard import GuardConfig

__all__ = ["HubMediaConfig", "_default_outbound_filter"]

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_HUB_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _default_outbound_filter(text: str) -> str | None:
    return text


def _validate_identifier(value: str, *, field: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"invalid {field}: {value!r}")
    if ".." in value:
        raise ValueError(f"invalid {field}: {value!r}")
    return value


def _validate_space_prefix(value: str) -> str:
    if not value:
        raise ValueError("space_prefix must not be empty")
    if ":" in value or "/" in value or any(ord(c) < 32 for c in value):
        raise ValueError(f"invalid space_prefix: {value!r}")
    return value


def _validate_hub_url(value: str) -> str:
    if not _HUB_URL_RE.match(value):
        raise ValueError(f"invalid hub_url: {value!r}")
    return value.rstrip("/")


def _validate_rooms(rooms: tuple[str, ...]) -> tuple[str, ...]:
    if not rooms:
        raise ValueError("rooms must not be empty")
    out: list[str] = []
    seen: set[str] = set()
    for room in rooms:
        _validate_identifier(room, field="room")
        if room in seen:
            raise ValueError(f"duplicate room: {room!r}")
        seen.add(room)
        out.append(room)
    return tuple(out)


@dataclass(frozen=True)
class HubMediaConfig(MediaConfig):
    """Connect a host to a remote hub (poll + post over HTTP)."""

    hub_url: str = field(kw_only=True)
    host_id: str = field(kw_only=True)
    token_env: str = field(kw_only=True)
    rooms: tuple[str, ...] = field(kw_only=True)
    cursor_dir: Path | None = field(default=None, kw_only=True)
    poll_interval_sec: float = field(default=2.0, kw_only=True)
    space_prefix: str = field(default="hub.", kw_only=True)
    guard: GuardConfig = field(default_factory=GuardConfig, kw_only=True)
    outbound_filter: Callable[[str], str | None] | None = field(
        default=None,
        hash=False,
        compare=False,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "hub_url", _validate_hub_url(self.hub_url))
        _validate_identifier(self.host_id, field="host_id")
        if not isinstance(self.token_env, str) or not self.token_env:
            raise ValueError("token_env must be a non-empty string")
        object.__setattr__(self, "rooms", _validate_rooms(self.rooms))
        object.__setattr__(self, "space_prefix", _validate_space_prefix(self.space_prefix))
        if self.poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be positive")
        if self.outbound_filter is None:
            object.__setattr__(self, "outbound_filter", _default_outbound_filter)

    @property
    def resolved_cursor_dir(self) -> Path:
        if self.cursor_dir is not None:
            return self.cursor_dir
        return self.state_dir / "hub-cursors"
