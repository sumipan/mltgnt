"""tests/test_react_pattern.py — Thought Phase-compatible unit testing."""
from __future__ import annotations

import logging

import pytest

from mltgnt.agent._parse import _parse_json_response
from mltgnt.agent._runner import AgentResult, AgentRunner


# ---------------------------------------------------------------------------
# _parse_json_response: thought Field Extraction
# ---------------------------------------------------------------------------


def test_parse_with_thought():
    raw = '{"thought": "Check the status", "tool": "search", "args": {"query": "today"}}'
    result = _parse_json_response(raw)
    assert result is not None
    assert result["thought"] == "Check the status"
    assert result["tool"] == "search"
    assert result["args"] == {"query": "today"}


def test_parse_without_thought_still_succeeds():
    raw = '{"tool": "search", "args": {"query": "test"}}'
    result = _parse_json_response(raw)
    assert result is not None
    assert result["tool"] == "search"
    assert "thought" not in result


def test_parse_without_thought_logs_warning(caplog):
    raw = '{"tool": "search", "args": {"query": "test"}}'
    with caplog.at_level(logging.WARNING, logger="mltgnt.agent._parse"):
        result = _parse_json_response(raw)
    assert result is not None
    assert any("thought" in msg.lower() for msg in caplog.messages)


def test_parse_thought_in_codeblock():
    raw = '```json\n{"thought": "Think", "tool": "done", "args": {}}\n```'
    result = _parse_json_response(raw)
    assert result is not None
    assert result["thought"] == "Think"


# ---------------------------------------------------------------------------
# AgentRunner: tool_trace Home thought Fields are recorded
# ---------------------------------------------------------------------------


def _make_runner(responses: list[str], tool_results: list[str]) -> AgentRunner:
    call_idx = {"n": 0}
    result_idx = {"n": 0}

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx >= len(responses):
            return None
        return responses[idx]

    def tool_executor(tool_name: str, tool_args: dict) -> str:
        idx = result_idx["n"]
        result_idx["n"] += 1
        if idx >= len(tool_results):
            return ""
        return tool_results[idx]

    return AgentRunner(
        llm_call=llm_call,
        tool_executor=tool_executor,
        terminal_tools=frozenset(["done"]),
        max_iterations=5,
    )


def test_trace_includes_thought():
    responses = [
        '{"thought": "First search", "tool": "search", "args": {"query": "q"}}',
        '{"thought": "Complete", "tool": "done", "args": {}}',
    ]
    runner = _make_runner(responses, ["Search"])
    result = runner.run("test")
    assert result is not None
    assert result.tool_trace is not None
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0]["thought"] == "First search"
    assert result.tool_trace[0]["tool"] == "search"
    assert result.tool_trace[0]["result"] == "Search"


def test_trace_thought_absent_when_missing(caplog):
    """thought processing continues even without response,tool_trace Home thought Keys are not included."""
    responses = [
        '{"tool": "search", "args": {"query": "q"}}',
        '{"tool": "done", "args": {}}',
    ]
    with caplog.at_level(logging.WARNING, logger="mltgnt.agent._parse"):
        runner = _make_runner(responses, ["Result"])
        result = runner.run("test")
    assert result is not None
    assert result.tool_trace is not None
    assert "thought" not in result.tool_trace[0]
    assert any("thought" in msg.lower() for msg in caplog.messages)


def test_trace_thought_only_present_when_provided():
    """thought In response thought key trace However, not included in non-response."""
    responses = [
        '{"thought": "Think", "tool": "search", "args": {}}',
        '{"tool": "search", "args": {}}',
        '{"tool": "done", "args": {}}',
    ]
    runner = _make_runner(responses, ["ok", "ok"])
    result = runner.run("test")
    assert result is not None
    assert result.tool_trace is not None
    assert result.tool_trace[0]["thought"] == "Think"
    assert "thought" not in result.tool_trace[1]
