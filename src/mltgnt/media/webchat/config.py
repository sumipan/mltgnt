"""WebChat-specific media settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mltgnt.media._core.config import MediaConfig

__all__ = ["WebChatMediaConfig"]


@dataclass(frozen=True)
class WebChatMediaConfig(MediaConfig):
    """Message store directory (required) and the local listen address of the single channel."""

    store_dir: Path = field(kw_only=True)
    host: str = "127.0.0.1"
    port: int = 8765
    space_id: str = "webchat"
