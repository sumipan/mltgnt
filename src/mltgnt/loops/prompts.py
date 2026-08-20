"""mltgnt.loops.prompts — LLM プロンプト生成・JSON 抽出・契約検証。"""
from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, cast

from mltgnt.bridges.llm_adapter import call_llm

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


@dataclass(frozen=True)
class ClarifyResponse:
    clear: bool
    question: str | None
    reason: str
    reasoning: str
    uncertain_flag: bool


@dataclass(frozen=True)
class DecomposeSubtask:
    id: str
    title: str
    kind: str
    prompt: str


@dataclass(frozen=True)
class DecomposeResponse:
    subtasks: list[DecomposeSubtask]
    reasoning: str
    uncertain_flag: bool


@dataclass(frozen=True)
class EvaluateResponse:
    achieved: bool
    score: int
    summary: str
    next_focus: str
    reasoning: str
    uncertain_flag: bool


@dataclass(frozen=True)
class LlmTrace:
    input: str
    raw_output: str
    parsed: dict[str, Any] | None
    reasoning: str
    config: dict[str, Any]
    metadata: dict[str, Any]
    uncertain_flag: bool
    error: str | None = None


def prompt_version() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out[:12] if out else "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def extract_json(text: str) -> dict[str, Any]:
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        candidate = fenced.group(1).strip()
        return cast(dict[str, Any], json.loads(candidate))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return cast(dict[str, Any], json.loads(text[start : end + 1]))
    raise ValueError("no JSON object found in LLM output")


def _validate_clarify(data: dict[str, Any]) -> ClarifyResponse:
    clear = data.get("clear")
    question = data.get("question")
    reason = data.get("reason", "")
    reasoning = data.get("reasoning", "")
    uncertain = bool(data.get("uncertain_flag", False))
    if not isinstance(clear, bool):
        raise ValueError("clear must be bool")
    if clear and question:
        raise ValueError("clear=true with non-empty question")
    if not clear:
        if not question or not str(question).strip():
            raise ValueError("clear=false requires non-empty question")
        question = str(question)
    else:
        question = None
    return ClarifyResponse(
        clear=clear,
        question=question,
        reason=str(reason),
        reasoning=str(reasoning),
        uncertain_flag=uncertain,
    )


def _validate_decompose(data: dict[str, Any], *, max_subtasks: int) -> DecomposeResponse:
    raw_tasks = data.get("subtasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("subtasks must be a list")
    if len(raw_tasks) == 0:
        raise ValueError("subtasks must not be empty")
    if len(raw_tasks) > max_subtasks:
        raise ValueError(f"too many subtasks: {len(raw_tasks)} > {max_subtasks}")
    seen: set[str] = set()
    subtasks: list[DecomposeSubtask] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            raise ValueError("subtask must be object")
        sid = str(item.get("id", ""))
        if not sid or sid in seen:
            raise ValueError(f"duplicate or empty subtask id: {sid!r}")
        seen.add(sid)
        kind = str(item.get("kind", ""))
        if kind not in ("auto", "human"):
            raise ValueError(f"invalid kind: {kind!r}")
        subtasks.append(
            DecomposeSubtask(
                id=sid,
                title=str(item.get("title", sid)),
                kind=kind,
                prompt=str(item.get("prompt", "")),
            )
        )
    return DecomposeResponse(
        subtasks=subtasks,
        reasoning=str(data.get("reasoning", "")),
        uncertain_flag=bool(data.get("uncertain_flag", False)),
    )


def _validate_evaluate(data: dict[str, Any]) -> EvaluateResponse:
    achieved = data.get("achieved")
    if not isinstance(achieved, bool):
        raise ValueError("achieved must be bool")
    score = data.get("score", 0)
    if not isinstance(score, int) or not (0 <= score <= 100):
        raise ValueError("score must be int 0..100")
    summary = str(data.get("summary", ""))
    next_focus = str(data.get("next_focus", ""))
    if not achieved and not next_focus.strip():
        raise ValueError("next_focus required when achieved=false")
    return EvaluateResponse(
        achieved=achieved,
        score=score,
        summary=summary,
        next_focus=next_focus,
        reasoning=str(data.get("reasoning", "")),
        uncertain_flag=bool(data.get("uncertain_flag", False)),
    )


def _call_with_retry(
    prompt: str,
    *,
    engine: str,
    model: str,
    validator: Callable[[dict[str, Any]], Any],
    retry_suffix: str = "\n\nReturn a JSON object only.",
) -> tuple[Any, LlmTrace]:
    version = prompt_version()
    start = time.monotonic()
    trace_id = f"loops-{int(time.time() * 1000)}"
    config = {"engine": engine, "model": model, "prompt_version": version}

    def _attempt(p: str) -> tuple[str, dict[str, Any] | None, str, str | None]:
        raw = call_llm(p, engine=engine, model=model)
        try:
            parsed = extract_json(raw)
            validator(parsed)
            reasoning = str(parsed.get("reasoning", ""))
            return raw, parsed, reasoning, None
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            return raw, None, "", str(exc)

    raw, parsed, reasoning, err = _attempt(prompt)
    if err is not None:
        raw2, parsed2, reasoning2, err2 = _attempt(prompt + retry_suffix)
        duration_ms = int((time.monotonic() - start) * 1000)
        if err2 is not None:
            trace = LlmTrace(
                input=prompt,
                raw_output=raw2,
                parsed=None,
                reasoning="",
                config=config,
                metadata={
                    "duration_ms": duration_ms,
                    "trace_id": trace_id,
                    "token_usage": None,
                    "retry": True,
                },
                uncertain_flag=False,
                error=err2,
            )
            raise ValueError(err2) from None
        result = validator(parsed2)  # type: ignore[arg-type]
        trace = LlmTrace(
            input=prompt,
            raw_output=raw2,
            parsed=parsed2,
            reasoning=reasoning2,
            config=config,
            metadata={
                "duration_ms": duration_ms,
                "trace_id": trace_id,
                "token_usage": None,
                "retry": True,
            },
            uncertain_flag=bool(parsed2.get("uncertain_flag", False)) if parsed2 else False,
        )
        return result, trace

    duration_ms = int((time.monotonic() - start) * 1000)
    result = validator(parsed)  # type: ignore[arg-type]
    trace = LlmTrace(
        input=prompt,
        raw_output=raw,
        parsed=parsed,
        reasoning=reasoning,
        config=config,
        metadata={"duration_ms": duration_ms, "trace_id": trace_id, "token_usage": None, "retry": False},
        uncertain_flag=bool(parsed.get("uncertain_flag", False)) if parsed else False,
    )
    return result, trace


def run_clarify(
    instruction: str,
    *,
    engine: str,
    model: str,
) -> tuple[ClarifyResponse, LlmTrace]:
    return _call_with_retry(instruction, engine=engine, model=model, validator=_validate_clarify)


def run_decompose(
    instruction: str,
    *,
    engine: str,
    model: str,
    max_subtasks: int,
) -> tuple[DecomposeResponse, LlmTrace]:
    return _call_with_retry(
        instruction,
        engine=engine,
        model=model,
        validator=lambda d: _validate_decompose(d, max_subtasks=max_subtasks),
    )


def run_evaluate(
    instruction: str,
    *,
    engine: str,
    model: str,
) -> tuple[EvaluateResponse, LlmTrace]:
    return _call_with_retry(instruction, engine=engine, model=model, validator=_validate_evaluate)


def build_clarify_instruction(body: str, *, round_num: int, max_rounds: int) -> str:
    return (
        f"You are clarifying an objective (round {round_num}/{max_rounds}).\n"
        f"Objective:\n{body}\n\n"
        "Respond with JSON: "
        '{"clear": bool, "question": str|null, "reason": str, '
        '"reasoning": str, "uncertain_flag": bool}'
    )


def build_decompose_instruction(body: str, *, iteration: int, max_subtasks: int, next_focus: str = "") -> str:
    extra = f"\nFocus for this iteration: {next_focus}\n" if next_focus else ""
    return (
        f"Decompose the objective into subtasks (iteration {iteration}). "
        f"Maximum {max_subtasks} subtasks.\n"
        f"Objective:\n{body}{extra}\n\n"
        "Respond with JSON: "
        '{"subtasks": [{"id": str, "title": str, "kind": "auto"|"human", "prompt": str}], '
        '"reasoning": str, "uncertain_flag": bool}'
    )


def build_evaluate_instruction(body: str, *, results_summary: str, iteration: int, max_iterations: int) -> str:
    return (
        f"Evaluate progress (iteration {iteration}/{max_iterations}).\n"
        f"Objective:\n{body}\n\nResults:\n{results_summary}\n\n"
        "Respond with JSON: "
        '{"achieved": bool, "score": int(0-100), "summary": str, "next_focus": str, '
        '"reasoning": str, "uncertain_flag": bool}'
    )
