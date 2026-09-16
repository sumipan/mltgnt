"""会話層の境界データ — mltgnt.interfaces.turn を re-export（#3317）。

型の二重定義を避け、くびれ契約の所有者を interfaces に一本化する。
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
