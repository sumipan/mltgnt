"""mltgnt.agent._runner — 汎用エージェントループ。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from mltgnt.agent._parse import _parse_json_response

_logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """エージェントの実行結果。"""
    tool: str
    args: dict[str, Any]
    raw_response: str
    tool_trace: list[dict] | None = None


class LLMCaller(Protocol):
    def __call__(
        self,
        prompt: str,
        *,
        tool_result: str | None = None,
    ) -> str | None: ...


class ToolExecutor(Protocol):
    def __call__(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> str: ...


class AgentRunner:
    """汎用エージェントループ。"""

    def __init__(
        self,
        *,
        llm_call: LLMCaller,
        tool_executor: ToolExecutor,
        terminal_tools: frozenset[str],
        max_iterations: int = 3,
        logger: logging.Logger | None = None,
    ) -> None:
        self._llm_call = llm_call
        self._tool_executor = tool_executor
        self._terminal_tools = terminal_tools
        self._max_iterations = max_iterations
        self._logger = logger or _logger

    def run(self, prompt: str) -> AgentResult | None:
        tool_trace: list[dict] = []
        tool_result: str | None = None

        for i in range(self._max_iterations):
            raw = self._llm_call(prompt, tool_result=tool_result)
            if raw is None:
                self._logger.warning("llm_call returned None at iteration %d", i)
                return None

            data = _parse_json_response(raw)
            if data is None:
                self._logger.warning("failed to parse JSON response at iteration %d: %r", i, raw)
                return None

            tool_name: str = data["tool"]
            args: dict = data["args"]

            if tool_name in self._terminal_tools:
                return AgentResult(
                    tool=tool_name,
                    args=args,
                    raw_response=raw,
                    tool_trace=tool_trace if tool_trace else None,
                )

            try:
                tool_result = self._tool_executor(tool_name, args)
            except Exception as exc:
                self._logger.error("tool_executor raised for tool %r: %s", tool_name, exc)
                return None

            tool_trace.append({"tool": tool_name, "args": args, "result": tool_result})

        self._logger.warning("max_iterations (%d) reached without terminal tool", self._max_iterations)
        return None
