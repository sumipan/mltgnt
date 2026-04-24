"""tests/agent/test_runner.py — AgentRunner の受け入れ条件テスト (#287)"""
import pytest
from mltgnt.agent import AgentRunner, AgentResult


# ---- helpers ----

def make_llm(responses: list):
    """呼び出されるたびに responses から順に返す llm_call モック。"""
    calls = iter(responses)

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        return next(calls)

    return llm_call


def make_executor(results: dict):
    """tool_name をキーに結果を返す tool_executor モック。"""
    def executor(tool_name: str, tool_args: dict) -> str:
        return results[tool_name]
    return executor


# ---- 正常系 ----

def test_terminal_tool_immediate():
    """#1: 終端ツールが即座に返る。"""
    runner = AgentRunner(
        llm_call=make_llm(['{"tool": "slack_reply", "args": {"message": "hello"}}']),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"slack_reply"}),
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.tool == "slack_reply"
    assert result.args == {"message": "hello"}


def test_two_step_non_terminal_then_terminal():
    """#2: 非終端→終端の2ステップ。tool_trace が記録される。"""
    runner = AgentRunner(
        llm_call=make_llm([
            '{"tool": "search", "args": {"q": "x"}}',
            '{"tool": "slack_reply", "args": {"message": "y"}}',
        ]),
        tool_executor=make_executor({"search": "found: y"}),
        terminal_tools=frozenset({"slack_reply"}),
        max_iterations=3,
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.tool == "slack_reply"
    assert result.args == {"message": "y"}
    assert result.tool_trace == [
        {"tool": "search", "args": {"q": "x"}, "result": "found: y"}
    ]


def test_json_in_code_block():
    """#3: コードブロック内 JSON が正しくパースされる。"""
    raw = '```json\n{"tool": "done", "args": {}}\n```'
    runner = AgentRunner(
        llm_call=make_llm([raw]),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"done"}),
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.tool == "done"


def test_args_key_omitted_backward_compat():
    """#4: args キー省略の後方互換。"""
    runner = AgentRunner(
        llm_call=make_llm(['{"tool": "slack_reply", "message": "hi"}']),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"slack_reply"}),
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.tool == "slack_reply"
    assert result.args == {"message": "hi"}


# ---- 異常系 ----

def test_llm_returns_none():
    """#5: LLM が None を返す → run() が None を返す。"""
    runner = AgentRunner(
        llm_call=make_llm([None]),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"slack_reply"}),
    )
    assert runner.run("prompt") is None


def test_json_parse_failure():
    """#6: JSON パース失敗 → run() が None を返す。"""
    runner = AgentRunner(
        llm_call=make_llm(["I don't know"]),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"slack_reply"}),
    )
    assert runner.run("prompt") is None


def test_missing_tool_key():
    """#7: tool キーなし → run() が None を返す。"""
    runner = AgentRunner(
        llm_call=make_llm(['{"action": "reply"}']),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"slack_reply"}),
    )
    assert runner.run("prompt") is None


def test_max_iterations_exceeded():
    """#8: max_iterations 超過（非終端ツールが3回連続）→ run() が None を返す。"""
    runner = AgentRunner(
        llm_call=make_llm([
            '{"tool": "search", "args": {"q": "a"}}',
            '{"tool": "search", "args": {"q": "b"}}',
            '{"tool": "search", "args": {"q": "c"}}',
        ]),
        tool_executor=make_executor({"search": "ok"}),
        terminal_tools=frozenset({"slack_reply"}),
        max_iterations=3,
    )
    assert runner.run("prompt") is None


def test_tool_executor_raises_exception():
    """#9: tool_executor が例外を送出 → ループ中断、run() が None を返す。"""
    def failing_executor(tool_name: str, tool_args: dict) -> str:
        raise RuntimeError("network error")

    runner = AgentRunner(
        llm_call=make_llm(['{"tool": "search", "args": {"q": "x"}}']),
        tool_executor=failing_executor,
        terminal_tools=frozenset({"slack_reply"}),
    )
    assert runner.run("prompt") is None


# ---- import 確認 ----

def test_import():
    """#11: mltgnt.agent から AgentRunner, AgentResult が import できる。"""
    from mltgnt.agent import AgentRunner, AgentResult  # noqa: F401
