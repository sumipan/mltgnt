"""mltgnt.agent.reflexion — deterministic default Reflexion evaluator (#3855)."""
from __future__ import annotations

import json
from typing import Any

from mltgnt.agent._runner import ReflexionVerdict

_ERROR_PREFIX = "[ERROR]"
_ERROR_EXCERPT_CHARS = 500


def _call_key(tool: object, args: object) -> tuple[str, str]:
    return str(tool), json.dumps(args, sort_keys=True, default=str)


class DefaultReflexionEvaluator:
    """Retry on ``[ERROR]`` results, host failure markers, or repeated calls."""

    def __init__(
        self,
        failure_markers: tuple[str, ...] = (),
        repeat_window: int = 3,
    ) -> None:
        self._failure_markers = failure_markers
        self._repeat_window = repeat_window

    def __call__(
        self,
        prompt: str,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: str,
        tool_trace: list[dict],
    ) -> ReflexionVerdict:
        if tool_result.lstrip().startswith(_ERROR_PREFIX):
            excerpt = tool_result.strip()[:_ERROR_EXCERPT_CHARS]
            return ReflexionVerdict(
                should_retry=True,
                feedback=(
                    "The tool returned an error. Read the error message, then change "
                    f"the arguments or choose a different tool. Error: {excerpt}"
                ),
            )

        for marker in self._failure_markers:
            if marker and marker in tool_result:
                return ReflexionVerdict(
                    should_retry=True,
                    feedback=(
                        f"The tool result contains the failure marker {marker!r}. "
                        "Fix the cause before continuing."
                    ),
                )

        if self._repeat_window > 0:
            key = _call_key(tool_name, tool_args)
            previous = tool_trace[:-1][-self._repeat_window:]
            if any(_call_key(e.get("tool"), e.get("args")) == key for e in previous):
                return ReflexionVerdict(
                    should_retry=True,
                    feedback=(
                        "You are repeating the same tool call with the same arguments. "
                        "Take a different approach."
                    ),
                )

        return ReflexionVerdict(should_retry=False, feedback="")
