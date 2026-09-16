"""会話層の境界データ（くびれ: TurnInput / TurnResult）と TurnHandler Protocol。

媒体非依存。媒体層の ID・ブロック表現をフィールドに持たない。
くびれの契約は mltgnt が所有し、媒体層はホストが実装する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class Attachment:
    """媒体非依存の添付参照。"""

    name: str
    content_type: str | None = None
    uri: str | None = None


@dataclass(frozen=True)
class HistoryMessage:
    """会話履歴の 1 メッセージ。"""

    role: str  # "user" | "assistant" | "system"
    text: str
    persona_id: str | None = None


@dataclass(frozen=True)
class TurnInput:
    """会話層が内側へ渡すくびれ入力。"""

    conversation_id: str
    text: str
    attachments: tuple[Attachment, ...] = ()
    history: tuple[HistoryMessage, ...] = ()
    persona_id: str | None = None


@dataclass(frozen=True)
class TurnResult:
    """会話層が外側へ返すくびれ出力。"""

    kind: Literal["reply", "task"]
    text: str = ""
    task_ref: str | None = None


@runtime_checkable
class TurnHandler(Protocol):
    """媒体層がくびれを呼ぶための契約。"""

    def handle(self, turn: TurnInput) -> TurnResult:
        """1 ターンを処理し、返信またはタスク参照を返す。"""
        ...
