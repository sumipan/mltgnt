"""mltgnt.interfaces.dispatch — ActDispatcher Protocol と ActResult。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ActResult:
    action: str
    success: bool
    detail: str


@runtime_checkable
class ActDispatcher(Protocol):
    def dispatch(self, action: str, args: dict[str, Any]) -> ActResult:
        """アクションを実行し結果を返す。"""
        ...
