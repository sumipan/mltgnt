"""mltgnt.interfaces.loops — loops ホスト境界の型と Protocol。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol

LoopStatus = Literal[
    "clarifying",
    "awaiting_answer",
    "decomposing",
    "replanning",
    "awaiting_plan_approval",
    "executing",
    "awaiting_human",
    "evaluating",
    "paused",
    "done",
    "failed",
    "cancelled",
]

StepStatus = Literal[
    "pending",
    "success",
    "failed_exit",
    "engine_error",
    "rejected",
    "empty_result",
    "other",
]

WatchStatus = Literal["pending", "satisfied", "failed"]


@dataclass(frozen=True)
class WatchVerdict:
    status: WatchStatus
    detail: str
    observed_token: str | None = None


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

    def post_progress(
        self,
        *,
        loop_id: str,
        persona: str,
        thread: HumanThreadRef,
        text: str,
        event_id: str,
    ) -> bool: ...

    def post_deliverable(
        self,
        *,
        loop_id: str,
        persona: str,
        thread: HumanThreadRef,
        deliverable_path: str,
        summary: str,
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


class ConditionEvaluator(Protocol):
    def evaluate(
        self,
        condition: Mapping[str, object],
        *,
        previous_token: str | None,
    ) -> WatchVerdict | None: ...


@dataclass(frozen=True)
class ActionRequest:
    name: str
    args: Mapping[str, object]


@dataclass(frozen=True)
class ActionResult:
    success: bool
    summary: str
    output: Mapping[str, object] | None = None


class ActionExecutor(Protocol):
    def execute(
        self, *, request: ActionRequest, idempotency_key: str
    ) -> ActionResult: ...


class MemoryAppender(Protocol):
    def __call__(
        self,
        *,
        persona: str,
        content: str,
        timestamp: str,
        dedupe_key: str,
    ) -> bool: ...
