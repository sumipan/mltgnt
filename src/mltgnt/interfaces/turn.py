"""Conversation-layer boundary types (neck: TurnInput / TurnResult) and TurnHandler Protocol.

Media-agnostic. No media-layer IDs or block shapes in fields.
mltgnt owns the neck contract; the host implements the media layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class Attachment:
    """Media-agnostic attachment reference."""

    name: str
    content_type: str | None = None
    uri: str | None = None


@dataclass(frozen=True)
class HistoryMessage:
    """One message in conversation history."""

    role: str  # "user" | "assistant" | "system"
    text: str
    persona_id: str | None = None


@dataclass(frozen=True)
class TurnInput:
    """Neck input from the conversation layer inward."""

    conversation_id: str
    text: str
    attachments: tuple[Attachment, ...] = ()
    history: tuple[HistoryMessage, ...] = ()
    persona_id: str | None = None


@dataclass(frozen=True)
class TurnResult:
    """Neck output from the conversation layer outward."""

    kind: Literal["reply", "task"]
    text: str = ""
    task_ref: str | None = None


@runtime_checkable
class TurnHandler(Protocol):
    """Contract for the media layer to call the neck."""

    def handle(self, turn: TurnInput) -> TurnResult:
        """Process one turn; return a reply or task reference."""
        ...
