"""Conversation-layer boundary types — re-export mltgnt.interfaces.turn (#3317).

Avoid duplicate type defs; neck contract ownership stays in interfaces.
"""

from __future__ import annotations

from mltgnt.interfaces.turn import (
    Attachment,
    HistoryMessage,
    TurnHandler,
    TurnInput,
    TurnResult,
)

__all__ = [
    "Attachment",
    "HistoryMessage",
    "TurnHandler",
    "TurnInput",
    "TurnResult",
]
