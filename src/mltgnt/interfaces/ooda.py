"""mltgnt.interfaces.ooda — OODA ループの Protocol とデータクラス。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ObservationEvent:
    event_id: str
    event_type: str
    status: str
    timestamp: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ActResult:
    action: str
    success: bool
    detail: str


@dataclass(frozen=True)
class OODAConfig:
    max_recovery_attempts: int = 3
    escalate_after: int = 2
    observe_filter: str | None = None


@dataclass(frozen=True)
class OODATickResult:
    observed_events: int
    actions_taken: list[ActResult]
    escalated: bool


@runtime_checkable
class ObserveSource(Protocol):
    def observe(self, *, since: str | None = None) -> list[ObservationEvent]:
        """前回観測以降の新規イベントを返す。since は ISO 8601 タイムスタンプ。"""
        ...


@runtime_checkable
class ActDispatcher(Protocol):
    def dispatch(self, action: str, args: dict[str, Any]) -> ActResult:
        """アクションを実行し結果を返す。"""
        ...
