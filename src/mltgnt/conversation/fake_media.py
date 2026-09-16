"""Fake 媒体層 — Slack 無しで TurnInput を組み立てる（#3317）。

会話層より内側のテスト・移行検証用。本番の入口には使わない。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mltgnt.conversation.types import Attachment, HistoryMessage, TurnInput


@dataclass(frozen=True)
class FakeMediaEvent:
    """媒体非依存の最小イベント。"""

    conversation_id: str
    text: str
    author: str = ""
    persona_id: str | None = None
    attachments: tuple[Attachment, ...] = ()
    history: tuple[HistoryMessage, ...] = field(default_factory=tuple)


def to_turn_input(event: FakeMediaEvent) -> TurnInput:
    """FakeMediaEvent → TurnInput（くびれ）。"""
    return TurnInput(
        conversation_id=event.conversation_id,
        text=event.text,
        attachments=event.attachments,
        history=event.history,
        persona_id=event.persona_id,
    )
