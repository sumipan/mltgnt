"""mltgnt.agent._runner — 汎用エージェントループ。"""
from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from mltgnt.agent._parse import _parse_json_response
from mltgnt.agent.action_classifier import ActionClassifier

_logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """エージェントの実行結果。"""
    tool: str
    args: dict[str, Any]
    raw_response: str
    tool_trace: list[dict] | None = None
    reflexion_count: int = 0


@dataclass
class ReflexionVerdict:
    """Reflexion 評価の判定結果。"""
    should_retry: bool
    feedback: str  # should_retry=True 時に LLM へ注入するフィードバック


class ReflexionEvaluator(Protocol):
    """ツール実行結果を評価し、再計画の要否を判定する。"""

    def __call__(
        self,
        prompt: str,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: str,
        tool_trace: list[dict],
    ) -> ReflexionVerdict: ...


@dataclass
class RetryConfig:
    """一時的障害に対するリトライ設定。"""
    max_retries: int = 2
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0


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
        max_iterations_fn: Callable[[str], int] | None = None,
        evaluator: ReflexionEvaluator | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
        audit_writer: Callable[[str, dict, str], None] | None = None,
        classifier: ActionClassifier | None = None,
    ) -> None:
        self._llm_call = llm_call
        self._tool_executor = tool_executor
        self._terminal_tools = terminal_tools
        self._max_iterations = max_iterations
        self._max_iterations_fn = max_iterations_fn
        self._evaluator = evaluator
        self._retry_config = retry_config
        self._logger = logger or _logger
        self._audit_writer = audit_writer
        self._classifier = classifier

    def _backoff_delay(self, attempt: int) -> float:
        config = self._retry_config
        assert config is not None
        return float(
            min(
                config.base_delay_s * (2 ** attempt) + random.uniform(0, 0.5),
                config.max_delay_s,
            )
        )

    def _call_llm_and_parse(
        self,
        prompt: str,
        tool_result: str | None,
        iteration: int,
    ) -> tuple[str, dict | list[dict]] | None:
        max_retries = self._retry_config.max_retries if self._retry_config else 0

        for attempt in range(max_retries + 1):
            raw = self._llm_call(prompt, tool_result=tool_result)
            if raw is None:
                self._logger.warning(
                    "llm_call returned None at iteration %d (attempt %d)",
                    iteration,
                    attempt,
                )
                if attempt < max_retries:
                    time.sleep(self._backoff_delay(attempt))
                    continue
                return None

            data = _parse_json_response(raw)
            if data is None:
                self._logger.warning(
                    "failed to parse JSON response at iteration %d (attempt %d): %r",
                    iteration,
                    attempt,
                    raw,
                )
                if attempt < max_retries:
                    time.sleep(self._backoff_delay(attempt))
                    continue
                return None

            return raw, data

        return None

    def _execute_tool_raw(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> tuple[str, Exception | None]:
        try:
            return self._tool_executor(tool_name, args), None
        except Exception as exc:
            self._logger.error("tool_executor raised for tool %r: %s", tool_name, exc)
            return "", exc

    def _process_tool_result(
        self,
        prompt: str,
        data: dict,
        executed_result: str,
        exc: Exception | None,
        tool_trace: list[dict],
        reflexion_count: int,
    ) -> tuple[str, int]:
        tool_name: str = data["tool"]
        args: dict = data["args"]

        if exc is not None:
            return f"{tool_name}: [ERROR] {exc}", reflexion_count

        classification: str | None = None
        if self._classifier is not None:
            classification = self._classifier.classify(tool_name, args).value

        trace_entry: dict = {"tool": tool_name, "args": args, "result": executed_result}
        if classification is not None:
            trace_entry["classification"] = classification
        if data.get("thought") is not None:
            trace_entry["thought"] = data["thought"]
        tool_trace.append(trace_entry)

        if self._audit_writer is not None:
            try:
                self._audit_writer(tool_name, args, executed_result)
            except Exception as audit_exc:
                self._logger.warning("audit_writer raised: %s", audit_exc)

        result_str = executed_result
        if self._evaluator is not None:
            verdict = self._evaluator(
                prompt, tool_name, args, executed_result, tool_trace
            )
            if verdict.should_retry:
                reflexion_count += 1
                result_str = f"[REFLEXION] {verdict.feedback}\n\n{executed_result}"

        return f"{tool_name}: {result_str}", reflexion_count

    def _run_parallel_tools(
        self,
        raw: str,
        tools: list[dict],
        tool_trace: list[dict],
        prompt: str,
        reflexion_count: int,
    ) -> tuple[str | None, AgentResult | None, int]:
        if not tools:
            return "", None, reflexion_count

        non_terminal = [t for t in tools if t["tool"] not in self._terminal_tools]
        terminal = [t for t in tools if t["tool"] in self._terminal_tools]

        result_lines: list[str] = []

        if non_terminal:
            with ThreadPoolExecutor(max_workers=len(non_terminal)) as executor:
                futures = [
                    (data, executor.submit(self._execute_tool_raw, data["tool"], data["args"]))
                    for data in non_terminal
                ]
                for data, future in futures:
                    executed_result, exc = future.result()
                    if exc is not None:
                        trace_entry: dict = {
                            "tool": data["tool"],
                            "args": data["args"],
                            "result": f"[ERROR] {exc}",
                        }
                        if data.get("thought") is not None:
                            trace_entry["thought"] = data["thought"]
                        if self._classifier is not None:
                            classification = self._classifier.classify(
                                data["tool"], data["args"]
                            ).value
                            trace_entry["classification"] = classification
                        tool_trace.append(trace_entry)
                        result_lines.append(f"{data['tool']}: [ERROR] {exc}")
                    else:
                        line, reflexion_count = self._process_tool_result(
                            prompt,
                            data,
                            executed_result,
                            None,
                            tool_trace,
                            reflexion_count,
                        )
                        result_lines.append(line)

        if terminal:
            t = terminal[0]
            return None, AgentResult(
                tool=t["tool"],
                args=t["args"],
                raw_response=raw,
                tool_trace=tool_trace if tool_trace else None,
                reflexion_count=reflexion_count,
            ), reflexion_count

        return "\n".join(result_lines), None, reflexion_count

    def run(self, prompt: str) -> AgentResult | None:
        effective_max = (
            self._max_iterations_fn(prompt)
            if self._max_iterations_fn
            else self._max_iterations
        )
        tool_trace: list[dict] = []
        tool_result: str | None = None
        reflexion_count = 0

        for i in range(effective_max):
            parsed = self._call_llm_and_parse(prompt, tool_result, i)
            if parsed is None:
                return None
            raw, data = parsed

            if isinstance(data, list):
                next_result, terminal_result, reflexion_count = self._run_parallel_tools(
                    raw, data, tool_trace, prompt, reflexion_count
                )
                if terminal_result is not None:
                    return terminal_result
                tool_result = next_result
                continue

            tool_name: str = data["tool"]
            args: dict = data["args"]

            if tool_name in self._terminal_tools:
                return AgentResult(
                    tool=tool_name,
                    args=args,
                    raw_response=raw,
                    tool_trace=tool_trace if tool_trace else None,
                    reflexion_count=reflexion_count,
                )

            try:
                executed_result = self._tool_executor(tool_name, args)
            except Exception as exc:
                self._logger.error("tool_executor raised for tool %r: %s", tool_name, exc)
                return None

            classification: str | None = None
            if self._classifier is not None:
                classification = self._classifier.classify(tool_name, args).value

            trace_entry: dict = {"tool": tool_name, "args": args, "result": executed_result}
            if classification is not None:
                trace_entry["classification"] = classification
            if data.get("thought") is not None:
                trace_entry["thought"] = data["thought"]
            tool_trace.append(trace_entry)

            if self._audit_writer is not None:
                try:
                    self._audit_writer(tool_name, args, executed_result)
                except Exception as exc:
                    self._logger.warning("audit_writer raised: %s", exc)

            if self._evaluator is not None:
                verdict = self._evaluator(
                    prompt, tool_name, args, executed_result, tool_trace
                )
                if verdict.should_retry:
                    reflexion_count += 1
                    tool_result = f"[REFLEXION] {verdict.feedback}\n\n{executed_result}"
                else:
                    tool_result = executed_result
            else:
                tool_result = executed_result

        self._logger.warning(
            "max_iterations (%d) reached without terminal tool", effective_max
        )
        return None
