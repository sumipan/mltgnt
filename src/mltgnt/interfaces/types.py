"""L1 DTO — type definitions for the interfaces layer.

No dependency on L3 (domain); keep type compatibility with L3
concrete classes via structural subtyping (Protocol).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable


class Message(TypedDict):
    """Chat message."""

    role: str
    content: str


@dataclass
class ChatInput:
    """Shared pipeline input for chat / Slack."""

    source: str
    session_key: str
    messages: list[Message]
    persona_name: str = ""
    model: str | None = None
    context_files: list[Path] = field(default_factory=list)
    context_memory_excerpt: str | None = None
    context_memory_preferences: str | None = None


@dataclass
class ChatOutput:
    """Pipeline output."""

    content: str
    persona_name: str
    timestamp: datetime
    session_key: str


@runtime_checkable
class PersonaFMBase(Protocol):
    """L1 Protocol for persona frontmatter. Only name is required."""

    name: str


@runtime_checkable
class ChatInputBase(Protocol):
    """L1 Protocol for chat pipeline input."""

    source: str
    session_key: str
    messages: list[Message]
    persona_name: str


@runtime_checkable
class ChatOutputBase(Protocol):
    """L1 Protocol for chat pipeline output."""

    content: str
    persona_name: str
    timestamp: datetime
    session_key: str
