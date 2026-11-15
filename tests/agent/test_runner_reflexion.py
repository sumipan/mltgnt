"""tests/agent/test_runner_reflexion.py — Reflexion / dynamic max_iterations / retry (#2085)"""
from __future__ import annotations

from unittest.mock import patch

from mltgnt.agent._runner import AgentRunner, RetryConfig


# ---- helpers ----

def make_llm(responses: list):
    """llm_call mock that returns responses in order."""
    calls = iter(responses)

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        return next(calls)

    return llm_call


def make_tracking_llm(responses: list):
    """llm_call mock that also records tool_result."""
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


# ---- Reflexion: evaluator unset ----

def test_no_evaluator_backward_compat():
    """Without evaluator, legacy behavior (no Reflexion)."""
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

# ---- dynamic max_iterations ----

def test_max_iterations_fn_used():
    """With max_iterations_fn: use prompt-dependent limit."""
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
    """Without max_iterations_fn: use self._max_iterations."""
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


# ---- exponential backoff retry ----

@patch("mltgnt.agent._runner.time.sleep")
@patch("mltgnt.agent._runner.random.uniform", return_value=0.0)
def test_retry_llm_none_all_fail(mock_uniform, mock_sleep):
    """With retry_config and continuous LLM None: return None."""
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
def test_backoff_capped_at_max_delay(mock_uniform, mock_sleep):
    """Backoff delay does not exceed max_delay_s."""
    runner = AgentRunner(
        llm_call=make_llm([None, None, None]),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"done"}),
        retry_config=RetryConfig(max_retries=2, base_delay_s=10.0, max_delay_s=15.0),
    )
    runner.run("prompt")
    for call_args in mock_sleep.call_args_list:
        assert call_args[0][0] <= 15.0
