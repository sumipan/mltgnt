"""mltgnt.loops.models — LoopState / Subtask / PendingQuestion の型と JSON 変換。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from mltgnt.interfaces.loops import HumanThreadRef, LoopStatus, StepSubmission

SCHEMA_VERSION = 1

TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "failed", "cancelled"})


@dataclass
class PendingQuestion:
    question_id: str
    text: str
    kind: str  # "clarify" | "human_subtask"

    def to_dict(self) -> dict[str, Any]:
        return {"question_id": self.question_id, "text": self.text, "kind": self.kind}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingQuestion:
        return cls(
            question_id=str(data["question_id"]),
            text=str(data["text"]),
            kind=str(data["kind"]),
        )


@dataclass
class Subtask:
    id: str
    title: str
    kind: str  # "auto" | "human"
    prompt: str
    status: str = "pending"  # pending | running | success | failed
    result: str = ""
    result_summary: str = ""
    result_filename: str = ""
    submission: StepSubmission | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "prompt": self.prompt,
            "status": self.status,
            "result": self.result,
            "result_summary": self.result_summary,
            "result_filename": self.result_filename,
        }
        if self.submission is not None:
            d["submission"] = {
                "uuid": self.submission.uuid,
                "result_filename": self.submission.result_filename,
                "submitted_at": self.submission.submitted_at,
                "reused": self.submission.reused,
            }
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Subtask:
        submission = None
        if data.get("submission"):
            s = data["submission"]
            submission = StepSubmission(
                uuid=str(s["uuid"]),
                result_filename=str(s["result_filename"]),
                submitted_at=str(s["submitted_at"]),
                reused=bool(s.get("reused", False)),
            )
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            kind=str(data["kind"]),
            prompt=str(data["prompt"]),
            status=str(data.get("status", "pending")),
            result=str(data.get("result", "")),
            result_summary=str(data.get("result_summary", "")),
            result_filename=str(data.get("result_filename", "")),
            submission=submission,
        )


@dataclass
class LoopState:
    loop_id: str
    objective_path: str
    objective_hash: str
    title: str
    body: str
    status: LoopStatus
    iteration: int
    max_iterations: int
    persona: str
    thread: HumanThreadRef | None = None
    clarify_round: int = 0
    pending_question: PendingQuestion | None = None
    subtasks: list[Subtask] = field(default_factory=list)
    current_subtask_id: str | None = None
    consecutive_errors: int = 0
    created_at: str = ""
    updated_at: str = ""
    schema_version: int = SCHEMA_VERSION
    delivered_events: dict[str, bool] = field(default_factory=dict)
    content_change_warning: str = ""
    next_focus: str = ""
    clarification_context: list[str] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = self.schema_version
        if self.thread is not None:
            d["thread"] = {"channel_id": self.thread.channel_id, "thread_ts": self.thread.thread_ts}
        else:
            d["thread"] = None
        if self.pending_question is not None:
            d["pending_question"] = self.pending_question.to_dict()
        else:
            d["pending_question"] = None
        d["subtasks"] = [s.to_dict() for s in self.subtasks]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopState:
        required_types: dict[str, type] = {
            "loop_id": str,
            "objective_path": str,
            "objective_hash": str,
            "title": str,
            "body": str,
            "status": str,
            "iteration": int,
            "max_iterations": int,
            "persona": str,
        }
        for key, expected in required_types.items():
            if key not in data or not isinstance(data[key], expected):
                raise ValueError(f"{key} must be {expected.__name__}")
            if expected is int and isinstance(data[key], bool):
                raise ValueError(f"{key} must be int, not bool")
        if data.get("schema_version", 1) != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {data.get('schema_version')}")
        if data["status"] not in {
            "clarifying", "awaiting_answer", "decomposing", "executing",
            "awaiting_human", "evaluating", "done", "failed", "cancelled",
        }:
            raise ValueError(f"invalid status: {data['status']!r}")
        if data["iteration"] < 0 or data["max_iterations"] < 0:
            raise ValueError("iteration values must be non-negative")

        thread = None
        if data.get("thread"):
            t = data["thread"]
            thread = HumanThreadRef(channel_id=str(t["channel_id"]), thread_ts=str(t["thread_ts"]))

        pending = None
        if data.get("pending_question"):
            pending = PendingQuestion.from_dict(data["pending_question"])

        subtasks = [Subtask.from_dict(s) for s in data.get("subtasks", [])]

        return cls(
            loop_id=str(data["loop_id"]),
            objective_path=str(data["objective_path"]),
            objective_hash=str(data["objective_hash"]),
            title=str(data["title"]),
            body=str(data["body"]),
            status=data["status"],
            iteration=int(data["iteration"]),
            max_iterations=int(data["max_iterations"]),
            persona=str(data["persona"]),
            thread=thread,
            clarify_round=int(data.get("clarify_round", 0)),
            pending_question=pending,
            subtasks=subtasks,
            current_subtask_id=data.get("current_subtask_id"),
            consecutive_errors=int(data.get("consecutive_errors", 0)),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            delivered_events=dict(data.get("delivered_events", {})),
            content_change_warning=str(data.get("content_change_warning", "")),
            next_focus=str(data.get("next_focus", "")),
            clarification_context=[str(item) for item in data.get("clarification_context", [])],
        )


def state_to_json(state: LoopState) -> str:
    return json.dumps(state.to_dict(), ensure_ascii=False, indent=2)


def state_from_json(text: str) -> LoopState:
    return LoopState.from_dict(json.loads(text))
