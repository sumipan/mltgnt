"""Media-agnostic inbound / outbound message types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mltgnt.interfaces.turn import Attachment

__all__ = ["MediaEvent", "OutboundMessage"]


@dataclass(frozen=True)
class MediaEvent:
    """One inbound message received from a medium."""

    space_id: str
    conversation_id: str
    message_id: str
    author: str
    text: str
    attachments: tuple[Attachment, ...] = ()
    raw: Any = None


@dataclass(frozen=True)
class OutboundMessage:
    """One message to send to a medium."""

    text: str
    space_id: str
    thread_id: str | None = None
