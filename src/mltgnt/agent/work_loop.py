"""mltgnt.agent.work_loop — generic work loop with optional skill delegation."""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from mltgnt.agent._parse import _parse_json_response
from mltgnt.agent._runner import (
    REFLEXION_EXHAUSTED_TOOL,
    AgentRunner,
    LLMCaller,
    RetryConfig,
    ToolExecutor,
)
from mltgnt.agent.plan import Plan, build_plan_prompt, parse_plan
from mltgnt.agent.reflexion import DefaultReflexionEvaluator
from mltgnt.bridges.ghdag_bridge import enqueue_and_wait
from mltgnt.interfaces.persona import PersonaProtocol
from mltgnt.interfaces.types import ChatInput, Message
from mltgnt.skill.loader import load as load_skill
from mltgnt.skill.models import SkillMeta
from mltgnt.skill import runner as skill_runner_mod

_logger = logging.getLogger(__name__)

FINISH_TOOL = "finish"
RUN_SKILL_TOOL = "run_skill"
_MAX_CONSECUTIVE_SAME_CALL = 2
_PLAN_PROMPT_MARKER = "Decompose the following task into a short ordered list"
_EVENT_ARG_MAX = 200
_EVENT_RESULT_MAX = 200


class WorkLoopDeadline(Exception):
    """Raised by the host ``llm_call`` when the work-loop time budget is exhausted."""


@dataclass(frozen=True)
class WorkLoopConfig:
    max_iterations: int = 40
    max_reflexions: int = 5
    tool_skills: tuple[str, ...] = ()


@dataclass
class WorkLoopOutcome:
    status: str
    reason: str
    message: str
    artifacts: list[str]
    plan: Plan | None
    trace: list[dict] | None


class SkillRunner(Protocol):
    def __call__(
        self,
        skill_name: str,
        arguments: str,
        *,
        parent_correlation_id: str | None,
    ) -> tuple[bool, str]: ...


def is_plan_prompt(prompt: str) -> bool:
    return _PLAN_PROMPT_MARKER in prompt


def _call_key(tool: str, args: dict[str, Any]) -> tuple[str, str]:
    return tool, json.dumps(args, sort_keys=True, default=str)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_plan_progress(plan: Plan | None) -> str:
    if plan is None:
        return ""
    lines: list[str] = []
    for item in plan.items:
        mark = "x" if item.status == "done" else " "
        lines.append(f"- [{mark}] {item.title}")
    return "\n".join(lines)


class RepeatGuard:
    """Block the third consecutive identical ``(tool, args)`` call."""

    def __init__(self, inner: ToolExecutor) -> None:
        self._inner = inner
        self._recent: list[tuple[str, str]] = []

    def __call__(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        key = _call_key(tool_name, tool_args)
        if (
            len(self._recent) >= _MAX_CONSECUTIVE_SAME_CALL
            and self._recent[-1] == key
            and self._recent[-2] == key
        ):
            return (
                f"[ERROR] {tool_name} was already called {_MAX_CONSECUTIVE_SAME_CALL} "
                f"times in a row with the same arguments; try a different approach."
            )
        self._recent.append(key)
        if len(self._recent) > _MAX_CONSECUTIVE_SAME_CALL:
            self._recent.pop(0)
        return self._inner(tool_name, tool_args)


class TrackingCaller:
    """Append plan progress to prompts and track LLM / iteration metadata."""

    def __init__(self, inner: LLMCaller, plan: Plan | None) -> None:
        self._inner = inner
        self._plan = plan
        self.last_ok: bool = True
        self.last_raw: str | None = None
        self.llm_calls: int = 0

    def __call__(
        self,
        prompt: str,
        *,
        tool_result: str | None = None,
    ) -> str | None:
        self.llm_calls += 1
        suffix = _format_plan_progress(self._plan)
        if suffix:
            prompt = f"{prompt}\n\n## Plan progress\n{suffix}"
        raw = self._inner(prompt, tool_result=tool_result)
        self.last_ok = raw is not None
        self.last_raw = raw
        return raw


def run_skill_contract(skills: Sequence[str]) -> str:
    if not skills:
        return ""
    allowed = ", ".join(skills)
    return (
        f'{RUN_SKILL_TOOL}: invoke a registered skill (allowed: {allowed}). '
        f'JSON args: {{"skill": "<name>", "arguments": "<optional text>"}}'
    )


def make_skill_tool(
    tools: ToolExecutor,
    runner: SkillRunner,
    *,
    allowed: frozenset[str],
    parent_correlation_id: str | None,
) -> ToolExecutor:
    def execute(tool_name: str, tool_args: dict[str, Any]) -> str:
        if tool_name != RUN_SKILL_TOOL:
            return tools(tool_name, tool_args)
        skill = tool_args.get("skill")
        if not isinstance(skill, str) or not skill.strip():
            return '[ERROR] run_skill needs a "skill" name'
        skill = skill.strip()
        if skill not in allowed:
            return f"[ERROR] skill not allowed: {skill}"
        arguments = tool_args.get("arguments", "")
        if arguments is None:
            arguments = ""
        if not isinstance(arguments, str):
            arguments = str(arguments)
        ok, body = runner(skill, arguments, parent_correlation_id=parent_correlation_id)
        if ok:
            return body
        return f"[ERROR] skill {skill} failed: {body}"

    return execute


class GhdagSkillRunner:
    """Run a skill via ``enqueue_and_wait`` (same flow as scheduler skill action)."""

    def __init__(
        self,
        skills: Mapping[str, SkillMeta],
        persona: PersonaProtocol,
        *,
        engine: str,
        model: str | None,
        jobs_dir: Path,
        exec_done_dir: Path,
        timeout: float,
        permission_by_skill: Mapping[str, str | None] | None = None,
        persona_dir: Path | None = None,
        enqueue: Callable[..., tuple[bool, str]] = enqueue_and_wait,
    ) -> None:
        self._skills = skills
        self._persona = persona
        self._engine = engine
        self._model = model
        self._jobs_dir = jobs_dir
        self._exec_done_dir = exec_done_dir
        self._timeout = timeout
        self._permission_by_skill = permission_by_skill or {}
        self._persona_dir = persona_dir
        self._enqueue = enqueue

    def __call__(
        self,
        skill_name: str,
        arguments: str,
        *,
        parent_correlation_id: str | None,
    ) -> tuple[bool, str]:
        meta = self._skills.get(skill_name)
        if meta is None:
            return False, f"skill not found: {skill_name}"
        skill_file = load_skill(meta)
        idempotency_key = f"work-loop:{skill_name}:{uuid4().hex}"
        session_key = parent_correlation_id or idempotency_key
        chat_input = ChatInput(
            source="work_loop",
            session_key=session_key,
            messages=[Message(role="user", content=arguments or "")],
            persona_name=self._persona.name,
            model=self._model,
        )
        run_output = skill_runner_mod.run(
            skill_file, self._persona, arguments, chat_input
        )
        prompt = next(
            m["content"]
            for m in run_output.chat_input.messages
            if m["role"] == "system"
        )
        resolved_model = run_output.chat_input.model
        permission = self._permission_by_skill.get(skill_name)
        return self._enqueue(
            prompt=prompt,
            engine=self._engine,
            model=resolved_model,
            timeout=self._timeout,
            idempotency_key=idempotency_key,
            jobs_dir=self._jobs_dir,
            exec_done_dir=self._exec_done_dir,
            persona_name=self._persona.name,
            persona_dir=self._persona_dir,
            parent_correlation_id=parent_correlation_id,
            permission=permission,
            run_result=run_output,
        )


def _emit_event(
    events_sink: Callable[[dict], None] | None,
    event: dict[str, Any],
) -> None:
    if events_sink is None:
        return
    try:
        events_sink(event)
    except Exception as exc:
        _logger.warning("events_sink raised: %s", exc)


def _blocked_outcome(
    reason: str,
    message: str,
    plan: Plan | None,
    trace: list[dict] | None,
) -> WorkLoopOutcome:
    return WorkLoopOutcome(
        status="BLOCKED",
        reason=reason,
        message=message,
        artifacts=[],
        plan=plan,
        trace=trace,
    )


def run_work_loop(
    order_text: str,
    *,
    llm_call: LLMCaller,
    tools: ToolExecutor,
    cfg: WorkLoopConfig,
    events_sink: Callable[[dict], None] | None = None,
    skill_runner: SkillRunner | None = None,
    parent_correlation_id: str | None = None,
) -> WorkLoopOutcome:
    plan: Plan | None = None
    try:
        plan_raw = llm_call(build_plan_prompt(order_text))
    except WorkLoopDeadline:
        return _blocked_outcome(
            "deadline_before_plan",
            "Work loop stopped before planning: deadline exceeded.",
            None,
            None,
        )

    if plan_raw is None:
        _emit_event(
            events_sink,
            {"type": "work_loop_plan_failed", "ts": time.time()},
        )
    else:
        try:
            plan = parse_plan(plan_raw)
        except ValueError:
            _emit_event(
                events_sink,
                {"type": "work_loop_plan_failed", "ts": time.time()},
            )
            plan = None

    executor: ToolExecutor = tools
    if skill_runner is not None:
        executor = make_skill_tool(
            tools,
            skill_runner,
            allowed=frozenset(cfg.tool_skills),
            parent_correlation_id=parent_correlation_id,
        )
    guarded = RepeatGuard(executor)
    tracking = TrackingCaller(llm_call, plan)
    step_counter = 0

    def step_hook(entry: dict) -> None:
        nonlocal step_counter
        step_counter += 1
        args_repr = json.dumps(entry.get("args"), default=str)
        _emit_event(
            events_sink,
            {
                "type": "work_loop_step",
                "ts": time.time(),
                "step": step_counter,
                "tool": entry.get("tool"),
                "args": _truncate(args_repr, _EVENT_ARG_MAX),
                "result_head": _truncate(str(entry.get("result", "")), _EVENT_RESULT_MAX),
                "plan": plan.progress() if plan else None,
            },
        )

    runner = AgentRunner(
        llm_call=tracking,
        tool_executor=guarded,
        terminal_tools=frozenset({FINISH_TOOL}),
        max_iterations=cfg.max_iterations,
        evaluator=DefaultReflexionEvaluator(),
        retry_config=RetryConfig(),
        history_mode="full_trace",
        plan=plan,
        max_reflexions=cfg.max_reflexions,
        step_hook=step_hook,
    )

    try:
        result = runner.run(order_text)
    except WorkLoopDeadline:
        return _blocked_outcome(
            "deadline",
            "Work loop stopped during execution: deadline exceeded.",
            plan,
            None,
        )

    if result is not None and result.tool == FINISH_TOOL:
        step_counter += 1
        args_repr = json.dumps(result.args, default=str)
        _emit_event(
            events_sink,
            {
                "type": "work_loop_step",
                "ts": time.time(),
                "step": step_counter,
                "tool": FINISH_TOOL,
                "args": _truncate(args_repr, _EVENT_ARG_MAX),
                "result_head": "",
                "plan": plan.progress() if plan else None,
            },
        )

    if result is None:
        if not tracking.last_ok:
            return WorkLoopOutcome(
                status="IMPL_FAILED",
                reason="llm_failed",
                message="LLM call failed during the work loop.",
                artifacts=[],
                plan=plan,
                trace=None,
            )
        # llm_calls also counts retries, so classify by the last response instead.
        if tracking.last_raw is None or _parse_json_response(tracking.last_raw) is None:
            return _blocked_outcome(
                "unparseable_response",
                "Could not parse the LLM response as a tool call.",
                plan,
                None,
            )
        return _blocked_outcome(
            "max_iterations",
            f"Work loop reached max_iterations ({cfg.max_iterations}).",
            plan,
            None,
        )

    if result.tool == REFLEXION_EXHAUSTED_TOOL:
        return _blocked_outcome(
            "reflexion_exhausted",
            "Reflexion retry limit was exceeded.",
            plan,
            result.tool_trace,
        )

    if result.tool == FINISH_TOOL:
        args = result.args
        message = args.get("message", "")
        if not isinstance(message, str):
            message = str(message)
        artifacts = args.get("artifacts", [])
        if not isinstance(artifacts, list):
            artifacts = []
        else:
            artifacts = [str(a) for a in artifacts]
        return WorkLoopOutcome(
            status="IMPL_DONE",
            reason="finished",
            message=message,
            artifacts=artifacts,
            plan=result.plan,
            trace=result.tool_trace,
        )

    return _blocked_outcome(
        "unparseable_response",
        "Agent stopped on an unexpected terminal tool.",
        plan,
        result.tool_trace,
    )
