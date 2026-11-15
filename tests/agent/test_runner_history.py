"""tests/agent/test_runner_history.py -- history_mode / _format_trace (#3855)."""
from __future__ import annotations

import json

from mltgnt.agent._runner import AgentRunner, _format_trace


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
