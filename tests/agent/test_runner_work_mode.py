"""tests/agent/test_runner_work_mode.py -- plan / max_reflexions / step_hook (#3855)."""
from __future__ import annotations

import json

from mltgnt.agent._runner import REFLEXION_EXHAUSTED_TOOL, AgentRunner
from mltgnt.agent.plan import Plan, PlanItem
from mltgnt.agent.reflexion import DefaultReflexionEvaluator


def make_tracking_llm(responses: list):
    calls = iter(responses)
    received: list[str | None] = []

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        received.append(tool_result)
        return next(calls)

    llm_call.received = received  # type: ignore[attr-defined]
    return llm_call


def _plan() -> Plan:
    return Plan(items=[PlanItem(id="a", title="A"), PlanItem(id="b", title="B")])


# ---- plan ----

def test_plan_update_on_terminal_response_is_applied():
    plan = _plan()
    llm = make_tracking_llm([
        json.dumps({
            "tool": "done", "args": {},
            "plan_update": [{"id": "a", "status": "done"}, {"id": "b", "status": "done"}],
        }),
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=lambda n, a: "ok",
        terminal_tools=frozenset({"done"}),
        plan=plan,
    )
    result = runner.run("prompt")
    assert result is not None and result.plan is not None
    assert result.plan.progress() == (2, 2)


def test_plan_none_ignores_plan_update():
    llm = make_tracking_llm([
        json.dumps({"tool": "work", "args": {}, "plan_update": [{"id": "a", "status": "done"}]}),
        json.dumps({"tool": "done", "args": {}}),
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=lambda n, a: "ok",
        terminal_tools=frozenset({"done"}),
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.plan is None
    assert llm.received == [None, "ok"]


# ---- max_reflexions ----

def _repeat_llm(n: int):
    return make_tracking_llm([json.dumps({"tool": "work", "args": {"x": 1}})] * n)


def test_max_reflexions_none_runs_to_max_iterations():
    llm = _repeat_llm(10)
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=lambda n, a: "[ERROR] boom",
        terminal_tools=frozenset({"done"}),
        evaluator=DefaultReflexionEvaluator(),
        max_iterations=10,
    )
    assert runner.run("prompt") is None
    assert len(llm.received) == 10


def test_max_reflexions_parallel_form():
    llm = make_tracking_llm([
        json.dumps({"tools": [
            {"tool": "w1", "args": {}},
            {"tool": "w2", "args": {}},
        ]}),
        json.dumps({"tool": "done", "args": {}}),
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=lambda n, a: "[ERROR] boom",
        terminal_tools=frozenset({"done"}),
        evaluator=DefaultReflexionEvaluator(),
        max_reflexions=1,
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.tool == REFLEXION_EXHAUSTED_TOOL
    assert result.reflexion_count == 2
    assert result.tool_trace is not None and len(result.tool_trace) == 2


# ---- step_hook ----

def test_step_hook_parallel_form_includes_error_entries():
    seen: list[dict] = []

    def executor(name: str, args: dict) -> str:
        if name == "bad":
            raise RuntimeError("kaboom")
        return "ok"

    llm = make_tracking_llm([
        json.dumps({"tools": [{"tool": "good", "args": {}}, {"tool": "bad", "args": {}}]}),
        json.dumps({"tool": "done", "args": {}}),
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=executor,
        terminal_tools=frozenset({"done"}),
        step_hook=seen.append,
    )
    result = runner.run("prompt")
    assert result is not None
    assert len(seen) == 2
    assert {e["tool"] for e in seen} == {"good", "bad"}
    assert any(e["result"].startswith("[ERROR]") for e in seen)


def test_step_hook_exception_does_not_stop_run():
    def hook(entry: dict) -> None:
        raise RuntimeError("hook failure")

    llm = make_tracking_llm([
        json.dumps({"tool": "t1", "args": {}}),
        json.dumps({"tool": "done", "args": {}}),
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=lambda n, a: "ok",
        terminal_tools=frozenset({"done"}),
        step_hook=hook,
    )
    result = runner.run("prompt")
    assert result is not None and result.tool == "done"


def test_single_form_executor_exception_still_returns_none():
    def executor(name: str, args: dict) -> str:
        raise RuntimeError("boom")

    seen: list[dict] = []
    llm = make_tracking_llm([json.dumps({"tool": "t1", "args": {}})])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=executor,
        terminal_tools=frozenset({"done"}),
        step_hook=seen.append,
    )
    assert runner.run("prompt") is None
    assert seen == []
