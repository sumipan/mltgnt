"""tests/agent/test_runner_reflexion.py — Reflexion / 動的 max_iterations / リトライ (#2085)"""
from __future__ import annotations

from unittest.mock import patch

from mltgnt.agent._runner import (
    AgentRunner,
    ReflexionVerdict,
    RetryConfig,
)


# ---- helpers ----

def make_llm(responses: list):
    """呼び出されるたびに responses から順に返す llm_call モック。"""
    calls = iter(responses)

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        return next(calls)

    return llm_call


def make_tracking_llm(responses: list):
    """tool_result 引数も記録する llm_call モック。"""
    calls = iter(responses)
    received: list[str | None] = []

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        received.append(tool_result)
        return next(calls)

    llm_call.received = received  # type: ignore[attr-defined]
    return llm_call


def make_executor(results: dict):
    def executor(tool_name: str, tool_args: dict) -> str:
        return results[tool_name]

    return executor


# ---- Reflexion: evaluator 未設定 ----

def test_no_evaluator_backward_compat():
    """evaluator 未設定時は従来動作（Reflexion 未発動）。"""
    llm = make_tracking_llm([
        '{"tool": "search", "args": {"q": "x"}}',
        '{"tool": "done", "args": {}}',
    ])
    runner = AgentRunner(
        llm_call=llm,
        tool_executor=make_executor({"search": "raw result"}),
        terminal_tools=frozenset({"done"}),
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.reflexion_count == 0
    assert llm.received[1] == "raw result"


# ---- Reflexion: should_retry=False ----

def test_evaluator_no_retry():
    """evaluator + should_retry=False: ツール結果がそのまま渡される。"""
    llm = make_tracking_llm([
        '{"tool": "search", "args": {"q": "x"}}',
        '{"tool": "done", "args": {}}',
    ])

    def evaluator(prompt, tool_name, tool_args, tool_result, tool_trace):
        return ReflexionVerdict(should_retry=False, feedback="unused")

    runner = AgentRunner(
        llm_call=llm,
        tool_executor=make_executor({"search": "raw result"}),
        terminal_tools=frozenset({"done"}),
        evaluator=evaluator,
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.reflexion_count == 0
    assert llm.received[1] == "raw result"


# ---- Reflexion: should_retry=True ----

def test_evaluator_retry_injects_feedback():
    """evaluator + should_retry=True: [REFLEXION] プレフィックス付きフィードバック注入。"""
    llm = make_tracking_llm([
        '{"tool": "search", "args": {"q": "x"}}',
        '{"tool": "done", "args": {}}',
    ])

    def evaluator(prompt, tool_name, tool_args, tool_result, tool_trace):
        return ReflexionVerdict(should_retry=True, feedback="try again")

    runner = AgentRunner(
        llm_call=llm,
        tool_executor=make_executor({"search": "raw result"}),
        terminal_tools=frozenset({"done"}),
        evaluator=evaluator,
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.reflexion_count == 1
    assert llm.received[1] == "[REFLEXION] try again\n\nraw result"


def test_reflexion_count_multiple():
    """reflexion_count が複数回の Reflexion 発動を正確に記録する。"""
    llm = make_tracking_llm([
        '{"tool": "search", "args": {"q": "a"}}',
        '{"tool": "search", "args": {"q": "b"}}',
        '{"tool": "done", "args": {}}',
    ])
    call_count = 0

    def evaluator(prompt, tool_name, tool_args, tool_result, tool_trace):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return ReflexionVerdict(should_retry=True, feedback=f"retry {call_count}")
        return ReflexionVerdict(should_retry=False, feedback="")

    runner = AgentRunner(
        llm_call=llm,
        tool_executor=make_executor({"search": "ok"}),
        terminal_tools=frozenset({"done"}),
        max_iterations=5,
        evaluator=evaluator,
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.reflexion_count == 2


# ---- 動的 max_iterations ----

def test_max_iterations_fn_used():
    """max_iterations_fn 設定時: プロンプトに応じた上限を使用。"""
    fn_calls: list[str] = []

    def max_fn(prompt: str) -> int:
        fn_calls.append(prompt)
        return 2 if "simple" in prompt else 8

    runner = AgentRunner(
        llm_call=make_llm([
            '{"tool": "search", "args": {"q": "a"}}',
            '{"tool": "search", "args": {"q": "b"}}',
            '{"tool": "search", "args": {"q": "c"}}',
        ]),
        tool_executor=make_executor({"search": "ok"}),
        terminal_tools=frozenset({"done"}),
        max_iterations=10,
        max_iterations_fn=max_fn,
    )
    assert runner.run("simple task") is None
    assert fn_calls == ["simple task"]

    search_response = '{"tool": "search", "args": {"q": "x"}}'
    runner2 = AgentRunner(
        llm_call=make_llm([search_response] * 8),
        tool_executor=make_executor({"search": "ok"}),
        terminal_tools=frozenset({"done"}),
        max_iterations=2,
        max_iterations_fn=max_fn,
    )
    assert runner2.run("complex task") is None
    assert fn_calls == ["simple task", "complex task"]


def test_max_iterations_fn_none_uses_default():
    """max_iterations_fn 未設定時: self._max_iterations を使用。"""
    runner = AgentRunner(
        llm_call=make_llm([
            '{"tool": "search", "args": {"q": "a"}}',
            '{"tool": "search", "args": {"q": "b"}}',
            '{"tool": "search", "args": {"q": "c"}}',
        ]),
        tool_executor=make_executor({"search": "ok"}),
        terminal_tools=frozenset({"done"}),
        max_iterations=2,
    )
    assert runner.run("prompt") is None


# ---- 指数バックオフ付きリトライ ----

@patch("mltgnt.agent._runner.time.sleep")
@patch("mltgnt.agent._runner.random.uniform", return_value=0.25)
def test_retry_llm_none(mock_uniform, mock_sleep):
    """retry_config 設定時・LLM が None: max_retries 回までリトライ。"""
    call_count = 0

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return None
        return '{"tool": "done", "args": {}}'

    runner = AgentRunner(
        llm_call=llm_call,
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"done"}),
        retry_config=RetryConfig(max_retries=2, base_delay_s=1.0, max_delay_s=30.0),
    )
    result = runner.run("prompt")
    assert result is not None
    assert call_count == 3
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(min(1.0 * (2 ** 0) + 0.25, 30.0))
    mock_sleep.assert_any_call(min(1.0 * (2 ** 1) + 0.25, 30.0))


@patch("mltgnt.agent._runner.time.sleep")
@patch("mltgnt.agent._runner.random.uniform", return_value=0.0)
def test_retry_llm_none_all_fail(mock_uniform, mock_sleep):
    """retry_config 設定時・LLM が None 連続: すべて失敗で None。"""
    runner = AgentRunner(
        llm_call=make_llm([None, None, None]),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"done"}),
        retry_config=RetryConfig(max_retries=2),
    )
    assert runner.run("prompt") is None
    assert mock_sleep.call_count == 2


@patch("mltgnt.agent._runner.time.sleep")
@patch("mltgnt.agent._runner.random.uniform", return_value=0.0)
def test_retry_parse_failure(mock_uniform, mock_sleep):
    """retry_config 設定時・パース失敗: max_retries 回までリトライ。"""
    call_count = 0

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return "not json"
        return '{"tool": "done", "args": {}}'

    runner = AgentRunner(
        llm_call=llm_call,
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"done"}),
        retry_config=RetryConfig(max_retries=2),
    )
    result = runner.run("prompt")
    assert result is not None
    assert call_count == 3
    assert mock_sleep.call_count == 2


def test_retry_no_retry_on_tool_executor_exception():
    """retry_config 設定時・tool_executor 例外: リトライせず即 None。"""

    def failing_executor(tool_name: str, tool_args: dict) -> str:
        raise RuntimeError("network error")

    llm_call_count = 0

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        nonlocal llm_call_count
        llm_call_count += 1
        return '{"tool": "search", "args": {"q": "x"}}'

    runner = AgentRunner(
        llm_call=llm_call,
        tool_executor=failing_executor,
        terminal_tools=frozenset({"done"}),
        retry_config=RetryConfig(max_retries=2),
    )
    with patch("mltgnt.agent._runner.time.sleep") as mock_sleep:
        assert runner.run("prompt") is None
        assert llm_call_count == 1
        mock_sleep.assert_not_called()


def test_no_retry_without_config():
    """retry_config 未設定時: リトライなし（従来動作）。"""
    call_count = 0

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        nonlocal call_count
        call_count += 1
        return None

    runner = AgentRunner(
        llm_call=llm_call,
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"done"}),
    )
    with patch("mltgnt.agent._runner.time.sleep") as mock_sleep:
        assert runner.run("prompt") is None
        assert call_count == 1
        mock_sleep.assert_not_called()


@patch("mltgnt.agent._runner.time.sleep")
@patch("mltgnt.agent._runner.random.uniform", return_value=0.0)
def test_backoff_capped_at_max_delay(mock_uniform, mock_sleep):
    """バックオフ遅延が max_delay_s を超えない。"""
    runner = AgentRunner(
        llm_call=make_llm([None, None, None]),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"done"}),
        retry_config=RetryConfig(max_retries=2, base_delay_s=10.0, max_delay_s=15.0),
    )
    runner.run("prompt")
    for call_args in mock_sleep.call_args_list:
        assert call_args[0][0] <= 15.0
