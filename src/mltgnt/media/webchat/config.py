"""WebChat-specific media settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from mltgnt.media._core.config import MediaConfig

__all__ = ["WebChatMediaConfig"]


@dataclass(frozen=True)
class WebChatMediaConfig(MediaConfig):
    """Message store directory (required) and the local listen address of the single channel."""

    store_dir: Path = field(kw_only=True)
    host: str = "127.0.0.1"
    port: int = 8765
    space_id: str = "webchat"
    assets_dir: Path | None = None
    avatars: Mapping[str, str] = field(default_factory=dict, hash=False)
    display_names: Mapping[str, str] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.avatars, MappingProxyType):
            object.__setattr__(self, "avatars", MappingProxyType(dict(self.avatars)))
        if not isinstance(self.display_names, MappingProxyType):
            object.__setattr__(self, "display_names", MappingProxyType(dict(self.display_names)))
