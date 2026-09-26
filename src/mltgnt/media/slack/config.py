"""Slack-specific media settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from mltgnt.interfaces.media import Status
from mltgnt.media._core.config import MediaConfig

__all__ = ["DEFAULT_STATUS_REACTIONS", "SlackMediaConfig"]

DEFAULT_STATUS_REACTIONS: Mapping[Status, str] = MappingProxyType(
    {
        Status.RECEIVED: "ok_woman",
        Status.WORKING: "woman-raising-hand",
        Status.DONE: "ok_woman",
        Status.FAILED: "x",
        Status.CANCELLED: "woman-gesturing-no",
    }
)


@dataclass(frozen=True)
class SlackMediaConfig(MediaConfig):
    """Token env var names, Status -> reaction name, and the per-message length limit."""

    bot_token_env: str = "SLACK_BOT_TOKEN"
    app_token_env: str = "SLACK_APP_TOKEN"
    status_reactions: Mapping[Status, str] = field(default_factory=lambda: DEFAULT_STATUS_REACTIONS, hash=False)
    chunk_max_chars: int = 3000

    def __post_init__(self) -> None:
        if not isinstance(self.status_reactions, MappingProxyType):
            object.__setattr__(self, "status_reactions", MappingProxyType(dict(self.status_reactions)))
        if self.chunk_max_chars <= 0:
            raise ValueError("chunk_max_chars must be positive")
