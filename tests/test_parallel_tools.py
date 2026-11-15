"""tests/test_parallel_tools.py — Acceptance tests for parallel tool execution (#1778)"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from mltgnt.agent._parse import _parse_json_response
from mltgnt.agent._runner import AgentRunner
from mltgnt.agent.action_classifier import ActionClass, ActionClassifier


# ---------------------------------------------------------------------------
# _parse_json_response: tools List format
# ---------------------------------------------------------------------------


def test_parse_tools_list():
    raw = json.dumps({
        "tools": [
            {"tool": "search", "args": {"q": "x"}},
            {"tool": "read", "args": {"path": "y"}},
        ]
    })
    result = _parse_json_response(raw)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["tool"] == "search"
    assert result[1]["tool"] == "read"


def test_parse_tools_empty_list():
    raw = '{"tools": []}'
    result = _parse_json_response(raw)
    assert result == []


def test_parse_tools_missing_tool_key_returns_none():
    raw = '{"tools": [{"args": {}}]}'
    assert _parse_json_response(raw) is None


def test_parse_tools_invalid_args_returns_none():
    raw = '{"tools": [{"tool": "search", "args": "bad"}]}'
    assert _parse_json_response(raw) is None


def test_parse_single_tool_still_returns_dict():
    raw = '{"tool": "search", "args": {"q": "x"}}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["tool"] == "search"


def test_parse_tools_with_thought_per_element():
    raw = json.dumps({
        "tools": [
            {"thought": "Search", "tool": "search", "args": {"q": "x"}},
            {"tool": "read", "args": {"path": "y"}},
        ]
    })
    result = _parse_json_response(raw)
    assert isinstance(result, list)
    assert result[0]["thought"] == "Search"
    assert "thought" not in result[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_llm(responses: list):
    calls = iter(responses)

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        return next(calls)

    return llm_call


def make_tracking_llm(responses: list):
    calls = iter(responses)
    received: list[str | None] = []

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        received.append(tool_result)
        return next(calls)

    llm_call.received = received  # type: ignore[attr-defined]
    return llm_call


def make_executor(results: dict, *, fail_tools: frozenset[str] = frozenset()):
    def executor(tool_name: str, tool_args: dict) -> str:
        if tool_name in fail_tools:
            raise RuntimeError(f"{tool_name} failed")
        return results.get(tool_name, "")

    return executor


# ---------------------------------------------------------------------------
# AgentRunner: Parallel execution
# ---------------------------------------------------------------------------


def test_parallel_empty_tools_list():
    """Home tools Lists are skipped and empty strings are LLM Contact Us"""
    llm = make_tracking_llm(['{"tools": []}', '{"tool": "done", "args": {}}'])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"done"}),
    )
    result = runner.run("prompt")
    assert result is not None
    assert llm.received[1] == ""


def test_parallel_terminal_tool_in_list():
    """When a terminal tool is in the list: run non-terminal first, then return AgentResult."""
    parallel_response = json.dumps({
        "tools": [
            {"tool": "search", "args": {"q": "x"}},
            {"tool": "slack_reply", "args": {"message": "hi"}},
        ]
    })
    executor_calls: list[str] = []

    def tracking_executor(tool_name: str, tool_args: dict) -> str:
        executor_calls.append(tool_name)
        return f"result-{tool_name}"

    runner = AgentRunner(
        llm_call=make_llm([parallel_response]),
        tool_executor=tracking_executor,
        terminal_tools=frozenset({"slack_reply"}),
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.tool == "slack_reply"
    assert result.args == {"message": "hi"}
    assert "search" in executor_calls
    assert "slack_reply" not in executor_calls
    assert result.tool_trace is not None
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0]["tool"] == "search"


def test_parallel_action_classifier(tmp_path: Path):
    """ActionClassifier is independent of each tool executed in classification """
    rules_path = tmp_path / "rules.jsonl"
    classifier = ActionClassifier(rules_path=rules_path)
    parallel_response = json.dumps({
        "tools": [
            {"tool": "search", "args": {"q": "x"}},
            {"tool": "read", "args": {"path": "y"}},
        ]
    })
    llm = make_tracking_llm([parallel_response, '{"tool": "done", "args": {}}'])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=make_executor({"search": "found", "read": "content"}),
        terminal_tools=frozenset({"done"}),
        classifier=classifier,
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.tool_trace is not None
    assert len(result.tool_trace) == 2
    for entry in result.tool_trace:
        assert entry["classification"] == ActionClass.NEEDS_REVIEW.value
