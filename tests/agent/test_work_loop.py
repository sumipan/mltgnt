"""Tests for mltgnt.agent.work_loop."""
from __future__ import annotations

import json
from typing import Any

from mltgnt.agent import build_plan_prompt
from mltgnt.agent.work_loop import (
    FINISH_TOOL,
    RepeatGuard,
    TrackingCaller,
    WorkLoopConfig,
    WorkLoopDeadline,
    is_plan_prompt,
    run_work_loop,
)


def _plan_json() -> str:
    return json.dumps(
        {
            "items": [
                {"id": "a", "title": "step one"},
                {"id": "b", "title": "step two", "depends": ["a"]},
            ]
        }
    )


def test_happy_path_with_events() -> None:
    responses = iter(
        [
            _plan_json(),
            '{"tool": "fetch", "args": {"url": "u"}}',
            '{"tool": "write", "args": {"path": "a.md"}}',
            '{"tool": "finish", "args": {"message": "ok", "artifacts": ["a.md"]}}',
        ]
    )

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        return next(responses)

    tool_calls: list[tuple[str, dict]] = []

    def tools(tool_name: str, tool_args: dict) -> str:
        tool_calls.append((tool_name, tool_args))
        return f"done:{tool_name}"

    events: list[dict] = []
    outcome = run_work_loop(
        "do work",
        llm_call=llm_call,
        tools=tools,
        cfg=WorkLoopConfig(),
        events_sink=events.append,
    )
    assert outcome.status == "IMPL_DONE"
    assert outcome.reason == "finished"
    assert outcome.message == "ok"
    assert outcome.artifacts == ["a.md"]
    step_events = [e for e in events if e["type"] == "work_loop_step"]
    assert len(step_events) == 3
    for ev in step_events:
        assert "step" in ev and "tool" in ev and "plan" in ev


def test_events_sink_none_and_failing_sink() -> None:
    def make_llm() -> Any:
        responses = iter(
            [
                _plan_json(),
                '{"tool": "finish", "args": {"message": "ok", "artifacts": []}}',
            ]
        )

        def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
            return next(responses)

        return llm_call

    outcome = run_work_loop(
        "x",
        llm_call=make_llm(),
        tools=lambda _t, _a: "",
        cfg=WorkLoopConfig(),
        events_sink=None,
    )
    assert outcome.status == "IMPL_DONE"

    def bad_sink(_event: dict) -> None:
        raise RuntimeError("sink failed")

    outcome2 = run_work_loop(
        "x",
        llm_call=make_llm(),
        tools=lambda _t, _a: "",
        cfg=WorkLoopConfig(),
        events_sink=bad_sink,
    )
    assert outcome2.status == "IMPL_DONE"


def test_deadline_before_and_during_plan() -> None:
    def raise_deadline(prompt: str, *, tool_result: str | None = None) -> str | None:
        raise WorkLoopDeadline()

    out = run_work_loop(
        "x",
        llm_call=raise_deadline,
        tools=lambda _t, _a: "",
        cfg=WorkLoopConfig(),
    )
    assert out.status == "BLOCKED"
    assert out.reason == "deadline_before_plan"

    calls = 0

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _plan_json()
        raise WorkLoopDeadline()

    out2 = run_work_loop(
        "x",
        llm_call=llm_call,
        tools=lambda _t, _a: "",
        cfg=WorkLoopConfig(),
    )
    assert out2.status == "BLOCKED"
    assert out2.reason == "deadline"


def test_plan_parse_failure_emits_event() -> None:
    events: list[dict] = []

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        if is_plan_prompt(prompt):
            return "not json"
        return '{"tool": "finish", "args": {"message": "ok", "artifacts": []}}'

    outcome = run_work_loop(
        "x",
        llm_call=llm_call,
        tools=lambda _t, _a: "",
        cfg=WorkLoopConfig(),
        events_sink=events.append,
    )
    assert outcome.status == "IMPL_DONE"
    assert [e["type"] for e in events if e["type"] == "work_loop_plan_failed"] == [
        "work_loop_plan_failed"
    ]


def test_max_iterations_and_llm_failed() -> None:
    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        if is_plan_prompt(prompt):
            return _plan_json()
        return '{"tool": "fetch", "args": {}}'

    out = run_work_loop(
        "x",
        llm_call=llm_call,
        tools=lambda _t, _a: "ok",
        cfg=WorkLoopConfig(max_iterations=2),
    )
    assert out.status == "BLOCKED"
    assert out.reason == "max_iterations"

    calls = 0

    def llm_none(prompt: str, *, tool_result: str | None = None) -> str | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _plan_json()
        return None

    out2 = run_work_loop(
        "x",
        llm_call=llm_none,
        tools=lambda _t, _a: "",
        cfg=WorkLoopConfig(),
    )
    assert out2.status == "IMPL_FAILED"
    assert out2.reason == "llm_failed"


def test_repeat_guard_blocks_third_identical_call() -> None:
    inner_calls: list[tuple[str, dict]] = []

    def inner(tool: str, args: dict) -> str:
        inner_calls.append((tool, args))
        return "ok"

    guard = RepeatGuard(inner)
    args = {"x": 1}
    assert guard("t", args) == "ok"
    assert guard("t", args) == "ok"
    blocked = guard("t", args)
    assert blocked.startswith("[ERROR]")
    assert len(inner_calls) == 2

    assert guard("other", {}) == "ok"
    assert guard("t", args) == "ok"


def test_is_plan_prompt() -> None:
    assert is_plan_prompt(build_plan_prompt("x")) is True
    assert is_plan_prompt("hello") is False


def test_tracking_caller_appends_plan() -> None:
    from mltgnt.agent.plan import Plan, PlanItem

    plan = Plan([PlanItem(id="1", title="alpha")])
    seen: list[str] = []

    def inner(prompt: str, *, tool_result: str | None = None) -> str | None:
        seen.append(prompt)
        return "{}"

    tc = TrackingCaller(inner, plan)
    tc("base", tool_result=None)
    assert "## Plan progress" in seen[0]
    assert "- [ ] alpha" in seen[0]
