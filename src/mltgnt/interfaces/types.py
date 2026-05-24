"""L1 DTO — interfaces 層の型定義。

L3 (domain) への依存を持たず、structural subtyping (Protocol) で
L3 具象クラスとの型互換を保つ。
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol, TypedDict, runtime_checkable


class Message(TypedDict):
    """チャットメッセージ。"""

    role: str
    content: str


@runtime_checkable
class PersonaFMBase(Protocol):
    """ペルソナフロントマターの L1 Protocol。name のみ必須。"""

    name: str


@runtime_checkable
class ChatInputBase(Protocol):
    """チャットパイプライン入力の L1 Protocol。"""

    source: str
    session_key: str
    messages: list[Message]
    persona_name: str


@runtime_checkable
class ChatOutputBase(Protocol):
    """チャットパイプライン出力の L1 Protocol。"""

    content: str
    persona_name: str
    timestamp: datetime
    session_key: str
