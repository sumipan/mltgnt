"""tests/agent/test_runner_history.py -- history_mode / _format_trace (#3855)."""
from __future__ import annotations

import json

from mltgnt.agent._runner import AgentRunner, ReflexionVerdict, _format_trace


def make_tracking_llm(responses: list):
    calls = iter(responses)
    received: list[str | None] = []

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        received.append(tool_result)
        return next(calls)

    llm_call.received = received  # type: ignore[attr-defined]
    return llm_call


def make_seq_executor(results: list[str]):
    it = iter(results)

    def executor(tool_name: str, tool_args: dict) -> str:
        return next(it)

    return executor


def _resp(tool: str, args: dict, thought: str | None = None) -> str:
    data: dict = {"tool": tool, "args": args}
    if thought is not None:
        data["thought"] = thought
    return json.dumps(data)


def test_full_trace_contains_all_steps():
    llm = make_tracking_llm([
        _resp("search", {"q": "x", "n": 1}, thought="look it up"),
        _resp("read", {"path": "p"}),
        _resp("write", {"path": "p"}),
        _resp("done", {}),
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=make_seq_executor(["result-1", "result-2", "result-3"]),
        terminal_tools=frozenset({"done"}),
        max_iterations=5,
        history_mode="full_trace",
    )
    result = runner.run("prompt")
    assert result is not None and result.tool == "done"
    third = llm.received[2]
    assert third is not None
    assert "## step 1: " in third
    assert "## step 2: " in third
    assert "search" in third and "read" in third
    assert json.dumps({"q": "x", "n": 1}, sort_keys=True) in third
    assert json.dumps({"path": "p"}, sort_keys=True) in third
    assert "result-1" in third and "result-2" in third
    assert "thought: look it up" in third
    assert "## step 3: " in llm.received[3]


def test_full_trace_truncates_oldest_but_not_latest():
    big = "A" * 300
    latest = "B" * 150
    llm = make_tracking_llm([
        _resp("search", {"q": "x"}),
        _resp("read", {"path": "p"}),
        _resp("done", {}),
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=make_seq_executor([big, latest]),
        terminal_tools=frozenset({"done"}),
        max_iterations=5,
        history_mode="full_trace",
        history_max_chars=200,
    )
    runner.run("prompt")
    third = llm.received[2]
    assert third is not None
    assert "[truncated 300 chars]" in third
    assert "## step 1: search" in third
    assert big not in third
    assert latest in third


def test_format_trace_keeps_headers_when_still_too_long():
    trace = [
        {"tool": "t1", "args": {}, "result": "x" * 50, "thought": "th"},
        {"tool": "t2", "args": {}, "result": "y" * 50},
    ]
    text = _format_trace(trace, 10)
    assert "## step 1: t1({})" in text
    assert "thought: th" in text
    assert "[truncated 50 chars]" in text
    assert "## step 2: t2({})" in text
    assert "y" * 50 in text


def test_format_trace_no_truncation_under_limit():
    trace = [{"tool": "t", "args": {"a": 1}, "result": "r"}]
    text = _format_trace(trace, 1000)
    assert text == '## step 1: t({"a": 1})\nr'


def test_full_trace_appends_reflexion_feedback():
    def evaluator(prompt, tool_name, tool_args, tool_result, tool_trace):
        return ReflexionVerdict(should_retry=True, feedback="try harder")

    llm = make_tracking_llm([
        _resp("search", {"q": "x"}),
        _resp("done", {}),
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=make_seq_executor(["r1"]),
        terminal_tools=frozenset({"done"}),
        evaluator=evaluator,
        history_mode="full_trace",
        history_max_chars=5,
    )
    result = runner.run("prompt")
    assert result is not None and result.reflexion_count == 1
    second = llm.received[1]
    assert second is not None
    assert second.rstrip("\n").splitlines()[-1] == "[REFLEXION] try harder"
    assert "## step 1: search" in second


def test_last_result_default_passes_only_previous_result():
    llm = make_tracking_llm([
        _resp("search", {"q": "x"}),
        _resp("read", {"path": "p"}),
        _resp("done", {}),
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=make_seq_executor(["result-1", "result-2"]),
        terminal_tools=frozenset({"done"}),
    )
    runner.run("prompt")
    assert llm.received == [None, "result-1", "result-2"]
