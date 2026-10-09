"""Public runner symbols on mltgnt.agent (#5037)."""
from __future__ import annotations

import importlib


def test_agent_runner_symbols_import_and_dunder_all() -> None:
    from mltgnt.agent import (  # noqa: F401
        LLMCaller,
        REFLEXION_EXHAUSTED_TOOL,
        RetryConfig,
        ToolExecutor,
    )

    import mltgnt.agent as agent

    for name in ("LLMCaller", "REFLEXION_EXHAUSTED_TOOL", "RetryConfig", "ToolExecutor"):
        assert name in agent.__all__

    runner = importlib.import_module("mltgnt.agent._runner")
    assert agent.LLMCaller is runner.LLMCaller
    assert agent.RetryConfig is runner.RetryConfig
    assert agent.ToolExecutor is runner.ToolExecutor
    assert agent.REFLEXION_EXHAUSTED_TOOL is runner.REFLEXION_EXHAUSTED_TOOL
    assert agent.REFLEXION_EXHAUSTED_TOOL == "__reflexion_exhausted__"
