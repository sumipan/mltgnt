"""tests/agent/test_runner.py — AgentRunner acceptance tests (#287)"""
from mltgnt.agent import AgentRunner


# ---- helpers ----

def make_llm(responses: list):
    """llm_call mock that returns responses in order."""
    calls = iter(responses)

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        return next(calls)

    return llm_call


def make_executor(results: dict):
    """tool_executor mock keyed by tool_name."""
    def executor(tool_name: str, tool_args: dict) -> str:
        return results[tool_name]
    return executor


# ---- happy path ----

def test_terminal_tool_immediate():
    """#1: terminal tool returns immediately."""
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
    """#2: non-terminal then terminal; tool_trace is recorded."""
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
    """#3: JSON inside a code block is parsed correctly."""
    raw = '```json\n{"tool": "done", "args": {}}\n```'
    runner = AgentRunner(
        llm_call=make_llm([raw]),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"done"}),
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.tool == "done"


def test_args_key_required():
    """v0.8.0: JSON without args key fails to parse."""
    runner = AgentRunner(
        llm_call=make_llm(['{"tool": "slack_reply", "message": "hi"}']),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"slack_reply"}),
    )
    assert runner.run("prompt") is None


# ---- error path ----

def test_llm_returns_none():
    """#5: LLM returns None → run() returns None."""
    runner = AgentRunner(
        llm_call=make_llm([None]),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"slack_reply"}),
    )
    assert runner.run("prompt") is None


def test_json_parse_failure():
    """#6: JSON parse failure → run() returns None."""
    runner = AgentRunner(
        llm_call=make_llm(["I don't know"]),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"slack_reply"}),
    )
    assert runner.run("prompt") is None


def test_missing_tool_key():
    """#7: missing tool key → run() returns None."""
    runner = AgentRunner(
        llm_call=make_llm(['{"action": "reply"}']),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"slack_reply"}),
    )
    assert runner.run("prompt") is None


def test_max_iterations_exceeded():
    """#8: max_iterations exceeded (3 non-terminal tools) → run() returns None."""
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
    """#9: tool_executor raises → loop stops, run() returns None."""
    def failing_executor(tool_name: str, tool_args: dict) -> str:
        raise RuntimeError("network error")

    runner = AgentRunner(
        llm_call=make_llm(['{"tool": "search", "args": {"q": "x"}}']),
        tool_executor=failing_executor,
        terminal_tools=frozenset({"slack_reply"}),
    )
    assert runner.run("prompt") is None


# ---- import checks ----

def test_import():
    """#11: AgentRunner, AgentResult importable from mltgnt.agent."""
    from mltgnt.agent import AgentRunner  # noqa: F401


# ---- audit_writer ----

def test_audit_writer_called_for_non_terminal_tool():
    """AC-1: audit_writer called once for a non-terminal tool."""
    calls = []

    def mock_writer(name, args, result):
        calls.append((name, args, result))

    runner = AgentRunner(
        llm_call=make_llm([
            '{"tool": "search", "args": {"q": "x"}}',
            '{"tool": "slack_reply", "args": {"message": "y"}}',
        ]),
        tool_executor=make_executor({"search": "found: y"}),
        terminal_tools=frozenset({"slack_reply"}),
        audit_writer=mock_writer,
    )
    runner.run("prompt")
    assert len(calls) == 1
    assert calls[0] == ("search", {"q": "x"}, "found: y")


def test_audit_writer_not_called_for_terminal_tool():
    """AC-2: audit_writer not called for terminal-only tools."""
    calls = []
    runner = AgentRunner(
        llm_call=make_llm(['{"tool": "slack_reply", "args": {"message": "hi"}}']),
        tool_executor=make_executor({}),
        terminal_tools=frozenset({"slack_reply"}),
        audit_writer=lambda n, a, r: calls.append((n, a, r)),
    )
    runner.run("prompt")
    assert calls == []


def test_audit_writer_none_default_compat():
    """AC-3: audit_writer=None (default) keeps existing tests passing."""
    runner = AgentRunner(
        llm_call=make_llm([
            '{"tool": "search", "args": {"q": "x"}}',
            '{"tool": "slack_reply", "args": {"message": "y"}}',
        ]),
        tool_executor=make_executor({"search": "found: y"}),
        terminal_tools=frozenset({"slack_reply"}),
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.tool == "slack_reply"


def test_audit_writer_exception_does_not_break_loop():
    """AC-4: audit_writer exceptions do not abort the loop."""
    def failing_writer(name, args, result):
        raise RuntimeError("audit write failed")

    runner = AgentRunner(
        llm_call=make_llm([
            '{"tool": "search", "args": {"q": "x"}}',
            '{"tool": "slack_reply", "args": {"message": "y"}}',
        ]),
        tool_executor=make_executor({"search": "found: y"}),
        terminal_tools=frozenset({"slack_reply"}),
        audit_writer=failing_writer,
    )
    result = runner.run("prompt")
    assert result is not None
    assert result.tool == "slack_reply"


def test_audit_writer_called_for_each_non_terminal_tool():
    """AC-5: audit_writer called on every non-terminal tool invocation."""
    calls = []
    runner = AgentRunner(
        llm_call=make_llm([
            '{"tool": "search", "args": {"q": "a"}}',
            '{"tool": "search", "args": {"q": "b"}}',
            '{"tool": "slack_reply", "args": {"message": "done"}}',
        ]),
        tool_executor=make_executor({"search": "ok"}),
        terminal_tools=frozenset({"slack_reply"}),
        max_iterations=5,
        audit_writer=lambda n, a, r: calls.append((n, a, r)),
    )
    runner.run("prompt")
    assert len(calls) == 2
    assert calls[0] == ("search", {"q": "a"}, "ok")
    assert calls[1] == ("search", {"q": "b"}, "ok")
