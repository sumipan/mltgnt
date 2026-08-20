"""mltgnt.interfaces.loops — loops ホスト境界の型と Protocol。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

LoopStatus = Literal[
    "clarifying",
    "awaiting_answer",
    "decomposing",
    "executing",
    "awaiting_human",
    "evaluating",
    "done",
    "failed",
    "cancelled",
]

StepStatus = Literal[
    "pending",
    "success",
    "failed_exit",
    "rejected",
    "empty_result",
    "other",
]


@dataclass(frozen=True)
class HumanThreadRef:
    channel_id: str
    thread_ts: str


@dataclass(frozen=True)
class StepSubmission:
    uuid: str
    result_filename: str
    submitted_at: str
    reused: bool


@dataclass(frozen=True)
class StepPoll:
    status: StepStatus
    content: str


class HumanChannel(Protocol):
    def open_thread(
        self,
        *,
        loop_id: str,
        persona: str,
        title: str,
        body: str,
        event_id: str,
    ) -> HumanThreadRef | None: ...

    def ask(
        self,
        *,
        loop_id: str,
        persona: str,
        thread: HumanThreadRef,
        question_id: str,
        text: str,
        event_id: str,
    ) -> bool: ...

    def notify(
        self,
        *,
        loop_id: str,
        persona: str,
        thread: HumanThreadRef,
        text: str,
        event_id: str,
    ) -> bool: ...

    def notify_fallback(
        self,
        *,
        loop_id: str,
        text: str,
        event_id: str,
    ) -> bool: ...

    def close_thread(
        self,
        *,
        loop_id: str,
        persona: str,
        thread: HumanThreadRef,
        event_id: str,
    ) -> bool: ...


class SubtaskExecutor(Protocol):
    def submit(
        self,
        *,
        prompt: str,
        idempotency_key: str,
        engine: str | None = None,
        model: str | None = None,
    ) -> StepSubmission: ...

    def poll(self, *, uuid: str, result_filename: str) -> StepPoll: ...
