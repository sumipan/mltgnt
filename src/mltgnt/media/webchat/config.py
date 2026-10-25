"""WebChat-specific media settings."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from mltgnt.media._core.config import MediaConfig

__all__ = ["WebChatMediaConfig"]


@dataclass(frozen=True)
class WebChatMediaConfig(MediaConfig):
    """Message store directory (required) and the local listen address of the single channel.

    ``display_names`` and ``avatars`` are keyed by message author (the inbound user is
    ``"user"``; a persona posts as its client ``author``). An avatar is a file name under
    ``assets_dir`` or an ``http(s)://`` URL. ``channel_name`` is shown in the UI header and composer. Message images live in
    ``files_dir`` (default: ``<store_dir>/../files``), at most ``max_upload_bytes`` each.
    """

    store_dir: Path = field(kw_only=True)
    host: str = "127.0.0.1"
    port: int = 8765
    space_id: str = "webchat"
    assets_dir: Path | None = None
    avatars: Mapping[str, str] = field(default_factory=dict, hash=False)
    display_names: Mapping[str, str] = field(default_factory=dict, hash=False)
    channel_name: str = "general"
    channel_slug: str = "general"
    files_dir: Path | None = None
    max_upload_bytes: int = 20 * 1024 * 1024

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9-]+", self.channel_slug):
            raise ValueError("channel_slug must match [a-z0-9-]+")
        if not isinstance(self.avatars, MappingProxyType):
            object.__setattr__(self, "avatars", MappingProxyType(dict(self.avatars)))
        if not isinstance(self.display_names, MappingProxyType):
            object.__setattr__(self, "display_names", MappingProxyType(dict(self.display_names)))

    @property
    def resolved_files_dir(self) -> Path:
        """``files_dir``, or ``files`` next to ``store_dir`` when unset."""
        return self.files_dir if self.files_dir is not None else Path(self.store_dir).parent / "files"
