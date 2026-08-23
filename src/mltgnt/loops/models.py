"""mltgnt.loops.models — LoopState / Subtask / PendingQuestion の型と JSON 変換。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from mltgnt.interfaces.loops import HumanThreadRef, LoopStatus, StepSubmission

SCHEMA_VERSION = 1

TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "failed", "cancelled"})

_VALID_STATUSES = frozenset({
    "clarifying",
    "awaiting_answer",
    "decomposing",
    "replanning",
    "awaiting_plan_approval",
    "executing",
    "awaiting_human",
    "evaluating",
    "done",
    "failed",
    "cancelled",
})


@dataclass
class PendingQuestion:
    question_id: str
    text: str
    kind: str  # "clarify" | "human_subtask" | "plan_approval"

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
    kind: str  # "auto" | "human" | "watch"
    prompt: str
    status: str = "pending"  # pending | running | success | failed
    result: str = ""
    result_summary: str = ""
    result_filename: str = ""
    submission: StepSubmission | None = None
    condition: dict[str, Any] | None = None
    depends: list[str] = field(default_factory=list)
    timeout_sec: int | None = None
    poll_interval_sec: int | None = None
    last_polled_at: str | None = None
    started_at: str | None = None
    watch_token: str | None = None

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
            "depends": list(self.depends),
        }
        if self.condition is not None:
            d["condition"] = dict(self.condition)
        if self.timeout_sec is not None:
            d["timeout_sec"] = self.timeout_sec
        if self.poll_interval_sec is not None:
            d["poll_interval_sec"] = self.poll_interval_sec
        if self.last_polled_at is not None:
            d["last_polled_at"] = self.last_polled_at
        if self.started_at is not None:
            d["started_at"] = self.started_at
        if self.watch_token is not None:
            d["watch_token"] = self.watch_token
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
        condition = data.get("condition")
        depends_raw = data.get("depends")
        if depends_raw is None:
            depends: list[str] = []
        else:
            depends = [str(x) for x in depends_raw]
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
            condition=dict(condition) if isinstance(condition, dict) else None,
            depends=depends,
            timeout_sec=int(data["timeout_sec"]) if data.get("timeout_sec") is not None else None,
            poll_interval_sec=(
                int(data["poll_interval_sec"])
                if data.get("poll_interval_sec") is not None
                else None
            ),
            last_polled_at=(
                str(data["last_polled_at"]) if data.get("last_polled_at") is not None else None
            ),
            started_at=str(data["started_at"]) if data.get("started_at") is not None else None,
            watch_token=str(data["watch_token"]) if data.get("watch_token") is not None else None,
        )


def apply_legacy_sequential_depends(subtasks: list[Subtask], *, had_depends_keys: bool) -> None:
    """旧 state（depends キー欠落）を逐次 depends に正規化する。"""
    if had_depends_keys or not subtasks:
        return
    for i, st in enumerate(subtasks):
        if i == 0:
            st.depends = []
        else:
            st.depends = [subtasks[i - 1].id]


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
    plan_approval: bool = True
    plan_revision: int = 0
    replan_count: int = 0
    replan_feedback: str = ""
    replan_trigger_subtask_id: str | None = None

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
        if data["status"] not in _VALID_STATUSES:
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

        raw_subtasks = data.get("subtasks", [])
        had_depends = any(isinstance(s, dict) and "depends" in s for s in raw_subtasks)
        subtasks = [Subtask.from_dict(s) for s in raw_subtasks]
        apply_legacy_sequential_depends(subtasks, had_depends_keys=had_depends)

        plan_approval = data.get("plan_approval", True)
        if not isinstance(plan_approval, bool):
            raise ValueError("plan_approval must be bool")

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
            plan_approval=plan_approval,
            plan_revision=int(data.get("plan_revision", 0)),
            replan_count=int(data.get("replan_count", 0)),
            replan_feedback=str(data.get("replan_feedback", "")),
            replan_trigger_subtask_id=data.get("replan_trigger_subtask_id"),
        )


def state_to_json(state: LoopState) -> str:
    return json.dumps(state.to_dict(), ensure_ascii=False, indent=2)


def state_from_json(text: str) -> LoopState:
    return LoopState.from_dict(json.loads(text))
