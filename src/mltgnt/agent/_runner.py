"""mltgnt.agent._runner — generic agent loop."""
from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from mltgnt.agent._parse import _parse_json_response
from mltgnt.agent.action_classifier import ActionClassifier
from mltgnt.agent.plan import Plan

_logger = logging.getLogger(__name__)

REFLEXION_EXHAUSTED_TOOL = "__reflexion_exhausted__"

HistoryMode = Literal["last_result", "full_trace"]


@dataclass
class AgentResult:
    """Result of an agent run."""
    tool: str
    args: dict[str, Any]
    raw_response: str
    tool_trace: list[dict] | None = None
    reflexion_count: int = 0
    plan: Plan | None = None


@dataclass
class ReflexionVerdict:
    """Verdict from a Reflexion evaluation."""
    should_retry: bool
    feedback: str  # feedback injected into the LLM when should_retry=True


class ReflexionEvaluator(Protocol):
    """Evaluate a tool result and decide whether to replan."""

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
    """Retry settings for transient failures."""
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


def _format_trace(trace: list[dict], max_chars: int) -> str:
    """Render ``tool_trace`` as numbered steps, folding old result bodies past ``max_chars``."""
    heads: list[str] = []
    bodies: list[str] = []
    for i, entry in enumerate(trace, start=1):
        args = json.dumps(entry.get("args"), ensure_ascii=False, sort_keys=True, default=str)
        head = f"## step {i}: {entry.get('tool')}({args})"
        if entry.get("thought") is not None:
            head += f"\nthought: {entry['thought']}"
        heads.append(head)
        bodies.append(str(entry.get("result", "")))

    def render() -> str:
        return "\n\n".join(f"{h}\n{b}" for h, b in zip(heads, bodies))

    text = render()
    for i in range(len(bodies) - 1):  # never fold the latest step
        if len(text) <= max_chars:
            break
        marker = f"[truncated {len(bodies[i])} chars]"
        if len(marker) < len(bodies[i]):
            bodies[i] = marker
            text = render()
    return text


class AgentRunner:
    """Generic agent loop."""

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
        history_mode: HistoryMode = "last_result",
        history_max_chars: int = 24_000,
        plan: Plan | None = None,
        max_reflexions: int | None = None,
        step_hook: Callable[[dict], None] | None = None,
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
        self._history_mode = history_mode
        self._history_max_chars = history_max_chars
        self._plan = plan
        self._max_reflexions = max_reflexions
        self._step_hook = step_hook

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

    def _apply_plan_update(self, data: dict) -> None:
        if self._plan is None or "plan_update" not in data:
            return
        self._plan.apply(data["plan_update"])

    def _append_trace(self, tool_trace: list[dict], entry: dict) -> None:
        tool_trace.append(entry)
        if self._step_hook is not None:
            try:
                self._step_hook(entry)
            except Exception as hook_exc:
                self._logger.warning("step_hook raised: %s", hook_exc)

    def _make_trace_entry(self, data: dict, result: str) -> dict:
        entry: dict = {"tool": data["tool"], "args": data["args"], "result": result}
        if self._classifier is not None:
            entry["classification"] = self._classifier.classify(
                data["tool"], data["args"]
            ).value
        if data.get("thought") is not None:
            entry["thought"] = data["thought"]
        return entry

    def _process_tool_result(
        self,
        prompt: str,
        data: dict,
        executed_result: str,
        tool_trace: list[dict],
        reflexion_count: int,
        feedbacks: list[str],
    ) -> tuple[str, int]:
        """Trace, audit and evaluate one successful tool call.

        Returns the result text for the next ``llm_call`` (``last_result`` mode)
        and the updated reflexion count. Retry feedback is appended to ``feedbacks``.
        """
        tool_name: str = data["tool"]
        args: dict = data["args"]

        self._append_trace(tool_trace, self._make_trace_entry(data, executed_result))

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
                feedbacks.append(verdict.feedback)
                result_str = f"[REFLEXION] {verdict.feedback}\n\n{executed_result}"

        return result_str, reflexion_count

    def _run_parallel_tools(
        self,
        raw: str,
        tools: list[dict],
        tool_trace: list[dict],
        prompt: str,
        reflexion_count: int,
        feedbacks: list[str],
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
                        self._append_trace(
                            tool_trace, self._make_trace_entry(data, f"[ERROR] {exc}")
                        )
                        result_lines.append(f"{data['tool']}: [ERROR] {exc}")
                    else:
                        result_str, reflexion_count = self._process_tool_result(
                            prompt,
                            data,
                            executed_result,
                            tool_trace,
                            reflexion_count,
                            feedbacks,
                        )
                        result_lines.append(f"{data['tool']}: {result_str}")

        if terminal:
            t = terminal[0]
            return None, AgentResult(
                tool=t["tool"],
                args=t["args"],
                raw_response=raw,
                tool_trace=tool_trace if tool_trace else None,
                reflexion_count=reflexion_count,
                plan=self._plan,
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
            feedbacks: list[str] = []

            if isinstance(data, list):
                for item in data:
                    self._apply_plan_update(item)
                next_result, terminal_result, reflexion_count = self._run_parallel_tools(
                    raw, data, tool_trace, prompt, reflexion_count, feedbacks
                )
                if terminal_result is not None:
                    return terminal_result
                tool_result = next_result
            else:
                self._apply_plan_update(data)
                tool_name: str = data["tool"]
                args: dict = data["args"]

                if tool_name in self._terminal_tools:
                    return AgentResult(
                        tool=tool_name,
                        args=args,
                        raw_response=raw,
                        tool_trace=tool_trace if tool_trace else None,
                        reflexion_count=reflexion_count,
                        plan=self._plan,
                    )

                executed_result, exc = self._execute_tool_raw(tool_name, args)
                if exc is not None:
                    return None

                tool_result, reflexion_count = self._process_tool_result(
                    prompt, data, executed_result, tool_trace, reflexion_count, feedbacks
                )

            if self._max_reflexions is not None and reflexion_count > self._max_reflexions:
                self._logger.warning(
                    "max_reflexions (%d) exceeded at iteration %d",
                    self._max_reflexions,
                    i,
                )
                return AgentResult(
                    tool=REFLEXION_EXHAUSTED_TOOL,
                    args={},
                    raw_response=raw,
                    tool_trace=tool_trace,
                    reflexion_count=reflexion_count,
                    plan=self._plan,
                )

            if self._history_mode == "full_trace":
                tool_result = _format_trace(tool_trace, self._history_max_chars)
                if feedbacks:
                    tool_result += "\n\n" + "\n".join(
                        f"[REFLEXION] {fb}" for fb in feedbacks
                    )

        self._logger.warning(
            "max_iterations (%d) reached without terminal tool", effective_max
        )
        return None
