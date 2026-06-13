"""tests/test_parallel_tools.py — 並列ツール実行の受け入れ条件テスト (#1778)"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mltgnt.agent._parse import _parse_json_response
from mltgnt.agent._runner import AgentRunner, ReflexionVerdict
from mltgnt.agent.action_classifier import ActionClass, ActionClassifier


# ---------------------------------------------------------------------------
# _parse_json_response: tools リスト形式
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
            {"thought": "検索する", "tool": "search", "args": {"q": "x"}},
            {"tool": "read", "args": {"path": "y"}},
        ]
    })
    result = _parse_json_response(raw)
    assert isinstance(result, list)
    assert result[0]["thought"] == "検索する"
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
# AgentRunner: 並列実行
# ---------------------------------------------------------------------------


@patch("mltgnt.agent._runner.ThreadPoolExecutor")
def test_parallel_tools_use_thread_pool(mock_executor_cls):
    """search と read が ThreadPoolExecutor 経由で並列実行される。"""
    mock_executor = mock_executor_cls.return_value.__enter__.return_value
    submitted: list[tuple[str, dict]] = []

    def capture_submit(fn, tool_name, args):
        submitted.append((tool_name, args))
        return _FakeFuture(fn(tool_name, args))

    mock_executor.submit.side_effect = capture_submit

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
        max_iterations=3,
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.tool == "done"
    mock_executor_cls.assert_called_once()
    assert mock_executor_cls.call_args.kwargs["max_workers"] == 2
    assert len(submitted) == 2
    assert submitted[0][0] == "search"
    assert submitted[1][0] == "read"


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


def test_parallel_three_tools_one_fails():
    """3 ツール並列で 1 つが例外の場合、残り 2 つは正常、失敗は [ERROR] 付き。"""
    parallel_response = json.dumps({
        "tools": [
            {"tool": "search", "args": {"q": "a"}},
            {"tool": "read", "args": {"path": "b"}},
            {"tool": "fetch", "args": {"url": "c"}},
        ]
    })
    llm = make_tracking_llm([parallel_response, '{"tool": "done", "args": {}}'])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=make_executor(
            {"search": "ok-search", "read": "ok-read", "fetch": "ok-fetch"},
            fail_tools=frozenset({"read"}),
        ),
        terminal_tools=frozenset({"done"}),
        max_iterations=3,
    )
    result = runner.run("prompt")
    assert result is not None
    tool_result = llm.received[1]
    assert "search: ok-search" in tool_result
    assert "fetch: ok-fetch" in tool_result
    assert "read: [ERROR]" in tool_result
    assert result.tool_trace is not None
    assert len(result.tool_trace) == 3
    assert result.tool_trace[1]["result"].startswith("[ERROR]")


def test_parallel_empty_tools_list():
    """空 tools リストはスキップされ、空文字列で次の LLM 呼び出しへ。"""
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
    """並列リスト内の terminal tool: 非 terminal を先に実行してから AgentResult を返す。"""
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


def test_parallel_tool_trace_format():
    """tool_trace の各エントリが既存フォーマットを維持する。"""
    parallel_response = json.dumps({
        "tools": [
            {"thought": "検索", "tool": "search", "args": {"q": "x"}},
            {"tool": "read", "args": {"path": "y"}},
        ]
    })
    llm = make_tracking_llm([parallel_response, '{"tool": "done", "args": {}}'])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=make_executor({"search": "found", "read": "content"}),
        terminal_tools=frozenset({"done"}),
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.tool_trace is not None
    assert result.tool_trace[0] == {
        "tool": "search",
        "args": {"q": "x"},
        "result": "found",
        "thought": "検索",
    }
    assert result.tool_trace[1] == {
        "tool": "read",
        "args": {"path": "y"},
        "result": "content",
    }


def test_parallel_reflexion_per_tool():
    """ReflexionEvaluator が各ツール結果に個別に評価される。"""
    parallel_response = json.dumps({
        "tools": [
            {"tool": "search", "args": {"q": "x"}},
            {"tool": "read", "args": {"path": "y"}},
        ]
    })
    llm = make_tracking_llm([parallel_response, '{"tool": "done", "args": {}}'])
    evaluated: list[str] = []

    def evaluator(prompt, tool_name, tool_args, tool_result, tool_trace):
        evaluated.append(tool_name)
        if tool_name == "search":
            return ReflexionVerdict(should_retry=True, feedback="retry search")
        return ReflexionVerdict(should_retry=False, feedback="")

    runner = AgentRunner(
        llm_call=llm,
        tool_executor=make_executor({"search": "raw-search", "read": "raw-read"}),
        terminal_tools=frozenset({"done"}),
        evaluator=evaluator,
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.reflexion_count == 1
    assert set(evaluated) == {"search", "read"}
    tool_result = llm.received[1]
    assert "search: [REFLEXION] retry search\n\nraw-search" in tool_result
    assert "read: raw-read" in tool_result


def test_parallel_action_classifier(tmp_path: Path):
    """ActionClassifier が並列実行された各ツールに独立して classification を記録。"""
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
