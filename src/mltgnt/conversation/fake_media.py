"""Fake media layer — build TurnInput without Slack (#3317).

For tests and migration checks inside the conversation layer. Not a production entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mltgnt.conversation.types import Attachment, HistoryMessage, TurnInput


@dataclass(frozen=True)
class FakeMediaEvent:
    """Minimal media-agnostic event."""

    conversation_id: str
    text: str
    author: str = ""
    persona_id: str | None = None
    attachments: tuple[Attachment, ...] = ()
    history: tuple[HistoryMessage, ...] = field(default_factory=tuple)


def to_turn_input(event: FakeMediaEvent) -> TurnInput:
    """FakeMediaEvent → TurnInput (neck)."""
    return TurnInput(
        conversation_id=event.conversation_id,
        text=event.text,
        attachments=event.attachments,
        history=event.history,
        persona_id=event.persona_id,
    )
