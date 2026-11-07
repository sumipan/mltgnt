"""Matrix-specific media settings."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlparse

from mltgnt.media._core.config import MediaConfig

__all__ = ["MatrixMediaConfig"]

_ROOM_KEY_RE = re.compile(r"[a-z0-9-]+")


def _validate_homeserver_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme == "https":
        if not parsed.netloc:
            raise ValueError(f"invalid homeserver_url: {value!r}")
        return value.rstrip("/")
    if parsed.scheme == "http":
        host = (parsed.hostname or "").lower()
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError(f"invalid homeserver_url: {value!r}")
        if not parsed.netloc:
            raise ValueError(f"invalid homeserver_url: {value!r}")
        return value.rstrip("/")
    raise ValueError(f"invalid homeserver_url: {value!r}")


def _validate_room_keys(rooms: Mapping[str, str]) -> MappingProxyType[str, str]:
    out: dict[str, str] = {}
    for key, room_id in rooms.items():
        if not _ROOM_KEY_RE.fullmatch(key):
            raise ValueError(f"invalid room key: {key!r}")
        if not isinstance(room_id, str) or not room_id:
            raise ValueError(f"invalid room id for key {key!r}")
        out[key] = room_id
    return MappingProxyType(out)


@dataclass(frozen=True)
class MatrixMediaConfig(MediaConfig):
    """Homeserver connection and room routing for the Matrix medium."""

    homeserver_url: str = field(kw_only=True)
    user_id: str = field(kw_only=True)
    access_token_env: str = "MATRIX_ACCESS_TOKEN"
    rooms: Mapping[str, str] = field(default_factory=dict, hash=False)
    allowed_senders: tuple[str, ...] = ()
    space_prefix: str = "matrix."
    files_dir: Path | None = None
    max_upload_bytes: int = 20 * 1024 * 1024
    poll_timeout_ms: int = 30000
    encrypted_room_notice: str = "I can't read encrypted rooms. Please talk to me in a room with encryption turned off."
    display_names: Mapping[str, str] = field(default_factory=dict, hash=False)
    avatars: Mapping[str, str] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "homeserver_url", _validate_homeserver_url(self.homeserver_url))
        if not self.user_id:
            raise ValueError("user_id must not be empty")
        if not self.space_prefix or ":" in self.space_prefix:
            raise ValueError(f"invalid space_prefix: {self.space_prefix!r}")
        if self.max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be positive")
        if self.poll_timeout_ms < 0:
            raise ValueError("poll_timeout_ms must not be negative")
        object.__setattr__(self, "rooms", _validate_room_keys(self.rooms))
        if not isinstance(self.display_names, MappingProxyType):
            object.__setattr__(self, "display_names", MappingProxyType(dict(self.display_names)))
        if not isinstance(self.avatars, MappingProxyType):
            object.__setattr__(self, "avatars", MappingProxyType(dict(self.avatars)))

    @property
    def resolved_files_dir(self) -> Path:
        return self.files_dir if self.files_dir is not None else Path(self.state_dir) / "files"

    def space_for_room(self, room_id: str) -> str | None:
        for key, rid in self.rooms.items():
            if rid == room_id:
                return f"{self.space_prefix}{key}"
        return None

    def room_for_space(self, space_id: str) -> str | None:
        if not space_id.startswith(self.space_prefix):
            return None
        key = space_id[len(self.space_prefix) :]
        return self.rooms.get(key)
