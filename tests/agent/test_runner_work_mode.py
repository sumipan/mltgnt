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

def test_plan_update_applied_and_returned():
    plan = _plan()
    llm = make_tracking_llm([
        json.dumps({
            "tool": "work", "args": {},
            "plan_update": [{"id": "a", "status": "done"}],
        }),
        json.dumps({"tool": "done", "args": {}}),
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=lambda n, a: "ok",
        terminal_tools=frozenset({"done"}),
        plan=plan,
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.plan is plan
    assert result.plan.progress() == (1, 2)


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


def test_plan_update_invalid_entries_ignored():
    plan = _plan()
    llm = make_tracking_llm([
        json.dumps({
            "tool": "work", "args": {},
            "plan_update": [
                {"id": "zzz", "status": "done"},
                {"id": "a", "status": "bogus"},
                {"id": "b", "status": "done"},
            ],
        }),
        json.dumps({"tool": "done", "args": {}, "plan_update": "not a list"}),
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=lambda n, a: "ok",
        terminal_tools=frozenset({"done"}),
        plan=plan,
    )
    result = runner.run("prompt")
    assert result is not None
    assert plan.progress() == (1, 2)
    assert plan.items[0].status == "pending"


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


def test_plan_update_parallel_form():
    plan = _plan()
    llm = make_tracking_llm([
        json.dumps({"tools": [
            {"tool": "w1", "args": {}, "plan_update": [{"id": "a", "status": "done"}]},
            {"tool": "w2", "args": {}, "plan_update": [{"id": "b", "status": "blocked"}]},
        ]}),
        json.dumps({"tool": "done", "args": {}}),
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=lambda n, a: "ok",
        terminal_tools=frozenset({"done"}),
        plan=plan,
    )
    result = runner.run("prompt")
    assert result is not None and result.plan is plan
    assert [i.status for i in plan.items] == ["done", "blocked"]


# ---- max_reflexions ----

def _repeat_llm(n: int):
    return make_tracking_llm([json.dumps({"tool": "work", "args": {"x": 1}})] * n)


def test_max_reflexions_exhausted_returns_result():
    plan = _plan()
    llm = _repeat_llm(10)
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=lambda n, a: "[ERROR] boom",
        terminal_tools=frozenset({"done"}),
        evaluator=DefaultReflexionEvaluator(),
        max_reflexions=2,
        max_iterations=10,
        plan=plan,
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.tool == REFLEXION_EXHAUSTED_TOOL == "__reflexion_exhausted__"
    assert result.args == {}
    assert result.reflexion_count == 3
    assert result.tool_trace is not None and len(result.tool_trace) == 3
    assert result.plan is plan
    assert result.raw_response == json.dumps({"tool": "work", "args": {"x": 1}})
    assert len(llm.received) == 3
    assert llm.received[1] is not None and "[REFLEXION]" in llm.received[1]


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

def test_step_hook_single_form():
    seen: list[dict] = []
    llm = make_tracking_llm([
        json.dumps({"tool": "t1", "args": {"i": 1}}),
        json.dumps({"tool": "t2", "args": {"i": 2}}),
        json.dumps({"tool": "t3", "args": {"i": 3}}),
        json.dumps({"tool": "done", "args": {}}),
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=lambda n, a: f"r-{n}",
        terminal_tools=frozenset({"done"}),
        max_iterations=5,
        step_hook=seen.append,
    )
    result = runner.run("prompt")
    assert result is not None and result.tool_trace is not None
    assert len(seen) == 3
    for hooked, entry in zip(seen, result.tool_trace):
        assert hooked is entry


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


def test_step_hook_sees_plan_update_already_applied():
    plan = _plan()
    statuses: list[str] = []

    def hook(entry: dict) -> None:
        statuses.append(plan.items[0].status)

    llm = make_tracking_llm([
        json.dumps({"tool": "t1", "args": {}, "plan_update": [{"id": "a", "status": "done"}]}),
        json.dumps({"tool": "done", "args": {}}),
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=lambda n, a: "ok",
        terminal_tools=frozenset({"done"}),
        plan=plan,
        step_hook=hook,
    )
    runner.run("prompt")
    assert statuses == ["done"]


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
