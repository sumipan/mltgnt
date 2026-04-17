from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypedDict


class Message(TypedDict):
    """チャットメッセージ。"""

    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class ChatInput:
    """チャット／Slack 共通のパイプライン入力（ホストの run_pipeline と対）。"""

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
    """パイプライン出力。"""

    content: str
    persona_name: str
    timestamp: datetime
    session_key: str
