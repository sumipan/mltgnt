"""mltgnt.loops.prompts — LLM プロンプト生成・JSON 抽出・契約検証。"""
from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, cast

from mltgnt.bridges.llm_adapter import call_llm

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

DEFAULT_WATCH_TIMEOUT_SEC = 14400
DEFAULT_WATCH_POLL_INTERVAL_SEC = 60
_WATCH_TIMEOUT_MIN = 60
_WATCH_TIMEOUT_MAX = 86400
_WATCH_POLL_MIN = 5
_WATCH_POLL_MAX = 3600

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
    condition: dict[str, object] | None = None
    depends: tuple[str, ...] = ()
    timeout_sec: int | None = None
    poll_interval_sec: int | None = None


@dataclass(frozen=True)
class DecomposeResponse:
    subtasks: list[DecomposeSubtask]
    reasoning: str
    uncertain_flag: bool


@dataclass(frozen=True)
class ReplanResponse:
    keep: tuple[str, ...]
    add: tuple[DecomposeSubtask, ...]
    reason: str
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


COMMENT_INTENTS = frozenset({"status", "instruction", "question", "chitchat"})


@dataclass(frozen=True)
class CommentClassifyResponse:
    intent: str
    reason: str
    reasoning: str
    uncertain_flag: bool


@dataclass(frozen=True)
class CommentReplyResponse:
    reply: str
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


class LlmCallError(ValueError):
    """失敗時のトレースを失わない LLM 呼び出しエラー。"""

    def __init__(self, message: str, trace: LlmTrace) -> None:
        super().__init__(message)
        self.trace = trace


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


def _parse_condition(item: Mapping[str, Any], *, kind: str) -> dict[str, object] | None:
    if kind == "watch":
        raw = item.get("condition")
        if not isinstance(raw, dict) or not raw:
            raise ValueError("watch requires non-empty condition object")
        if "type" not in raw:
            raise ValueError("condition.type is required")
        return {str(k): v for k, v in raw.items()}
    if "condition" in item and item["condition"] is not None:
        raise ValueError(f"{kind} must not include condition")
    return None


def _parse_timeout_poll(
    item: Mapping[str, Any], *, kind: str
) -> tuple[int | None, int | None]:
    if kind != "watch":
        if "timeout_sec" in item and item["timeout_sec"] is not None:
            raise ValueError("timeout_sec is only valid for watch")
        if "poll_interval_sec" in item and item["poll_interval_sec"] is not None:
            raise ValueError("poll_interval_sec is only valid for watch")
        return None, None

    timeout = item.get("timeout_sec", DEFAULT_WATCH_TIMEOUT_SEC)
    poll = item.get("poll_interval_sec", DEFAULT_WATCH_POLL_INTERVAL_SEC)
    if isinstance(timeout, bool) or not isinstance(timeout, int):
        raise ValueError("timeout_sec must be int")
    if isinstance(poll, bool) or not isinstance(poll, int):
        raise ValueError("poll_interval_sec must be int")
    if not (_WATCH_TIMEOUT_MIN <= timeout <= _WATCH_TIMEOUT_MAX):
        raise ValueError(f"timeout_sec out of range: {timeout}")
    if not (_WATCH_POLL_MIN <= poll <= _WATCH_POLL_MAX):
        raise ValueError(f"poll_interval_sec out of range: {poll}")
    return timeout, poll


def _parse_depends(
    item: Mapping[str, Any],
    *,
    previous_id: str | None,
) -> tuple[str, ...]:
    if "depends" not in item:
        return () if previous_id is None else (previous_id,)
    raw = item["depends"]
    if not isinstance(raw, list):
        raise ValueError("depends must be a list")
    depends: list[str] = []
    seen: set[str] = set()
    for dep in raw:
        if not isinstance(dep, str) or not dep:
            raise ValueError("depends entries must be non-empty strings")
        if dep in seen:
            raise ValueError(f"duplicate depends entry: {dep!r}")
        seen.add(dep)
        depends.append(dep)
    return tuple(depends)


def _detect_cycle(ids: list[str], depends_map: dict[str, tuple[str, ...]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ValueError(f"circular depends involving {node!r}")
        visiting.add(node)
        for dep in depends_map.get(node, ()):
            if dep not in depends_map and dep not in ids:
                continue
            dfs(dep)
        visiting.remove(node)
        visited.add(node)

    for sid in ids:
        dfs(sid)


def _validate_subtask_items(
    raw_tasks: list[Any],
    *,
    max_subtasks: int,
    known_ids: set[str] | None = None,
    allow_empty: bool = False,
) -> list[DecomposeSubtask]:
    if len(raw_tasks) == 0:
        if allow_empty:
            return []
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
        if known_ids is not None and sid in known_ids:
            raise ValueError(f"duplicate subtask id with keep: {sid!r}")
        seen.add(sid)
        kind = str(item.get("kind", ""))
        if kind not in ("auto", "human", "watch"):
            raise ValueError(f"invalid kind: {kind!r}")

        prompt_raw = item.get("prompt", "" if kind == "watch" else None)
        if kind == "watch":
            prompt = "" if prompt_raw is None else str(prompt_raw)
        else:
            if prompt_raw is None or not str(prompt_raw).strip():
                raise ValueError(f"{kind} requires non-empty prompt")
            prompt = str(prompt_raw)

        condition = _parse_condition(item, kind=kind)
        timeout_sec, poll_interval_sec = _parse_timeout_poll(item, kind=kind)
        previous_id = subtasks[-1].id if subtasks else None
        depends = _parse_depends(item, previous_id=previous_id)
        if sid in depends:
            raise ValueError(f"self-dependency not allowed: {sid!r}")

        subtasks.append(
            DecomposeSubtask(
                id=sid,
                title=str(item.get("title", sid)),
                kind=kind,
                prompt=prompt,
                condition=condition,
                depends=depends,
                timeout_sec=timeout_sec,
                poll_interval_sec=poll_interval_sec,
            )
        )

    id_set = {s.id for s in subtasks}
    if known_ids is not None:
        id_set |= known_ids
    depends_map = {s.id: s.depends for s in subtasks}
    for st in subtasks:
        for dep in st.depends:
            if dep not in id_set:
                raise ValueError(f"unknown depends id: {dep!r}")
    _detect_cycle([s.id for s in subtasks], depends_map)
    return subtasks


def _validate_decompose(data: dict[str, Any], *, max_subtasks: int) -> DecomposeResponse:
    raw_tasks = data.get("subtasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("subtasks must be a list")
    subtasks = _validate_subtask_items(raw_tasks, max_subtasks=max_subtasks)
    return DecomposeResponse(
        subtasks=subtasks,
        reasoning=str(data.get("reasoning", "")),
        uncertain_flag=bool(data.get("uncertain_flag", False)),
    )


def _validate_replan(
    data: dict[str, Any],
    *,
    existing_ids: set[str],
    required_keep: set[str],
    max_subtasks: int,
) -> ReplanResponse:
    keep_raw = data.get("keep")
    add_raw = data.get("add")
    if not isinstance(keep_raw, list):
        raise ValueError("keep must be a list")
    if not isinstance(add_raw, list):
        raise ValueError("add must be a list")

    keep: list[str] = []
    seen_keep: set[str] = set()
    for kid in keep_raw:
        if not isinstance(kid, str) or not kid:
            raise ValueError("keep entries must be non-empty strings")
        if kid not in existing_ids:
            raise ValueError(f"keep unknown id: {kid!r}")
        if kid in seen_keep:
            raise ValueError(f"duplicate keep id: {kid!r}")
        seen_keep.add(kid)
        keep.append(kid)

    missing_required = required_keep - seen_keep
    if missing_required:
        raise ValueError(
            f"running/success tasks must be kept: {sorted(missing_required)}"
        )

    add = _validate_subtask_items(
        add_raw,
        max_subtasks=max_subtasks,
        known_ids=seen_keep,
        allow_empty=True,
    )
    # empty add is allowed when keep alone is enough — but at least one of keep/add
    if not keep and not add:
        raise ValueError("replan keep+add must not both be empty")
    if len(keep) + len(add) > max_subtasks:
        raise ValueError(
            f"too many subtasks after replan: {len(keep) + len(add)} > {max_subtasks}"
        )

    # depends for add may only reference keep+add
    id_set = set(keep) | {s.id for s in add}
    depends_map = {s.id: s.depends for s in add}
    for kid in keep:
        depends_map.setdefault(kid, ())
    for st in add:
        for dep in st.depends:
            if dep not in id_set:
                raise ValueError(f"unknown depends id: {dep!r}")
    _detect_cycle(list(id_set), depends_map)

    return ReplanResponse(
        keep=tuple(keep),
        add=tuple(add),
        reason=str(data.get("reason", "")),
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


def _validate_comment_classify(data: dict[str, Any]) -> CommentClassifyResponse:
    intent = data.get("intent")
    if not isinstance(intent, str) or intent not in COMMENT_INTENTS:
        raise ValueError(
            f"intent must be one of {sorted(COMMENT_INTENTS)}, got {intent!r}"
        )
    return CommentClassifyResponse(
        intent=intent,
        reason=str(data.get("reason", "")),
        reasoning=str(data.get("reasoning", "")),
        uncertain_flag=bool(data.get("uncertain_flag", False)),
    )


def _validate_comment_reply(data: dict[str, Any]) -> CommentReplyResponse:
    reply = data.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("reply must be a non-empty string")
    return CommentReplyResponse(
        reply=reply.strip(),
        reasoning=str(data.get("reasoning", "")),
        uncertain_flag=bool(data.get("uncertain_flag", False)),
    )


def _unwrap_llm_result(result: Any) -> tuple[str, str | None]:
    """call_llm の戻り値から (stdout テキスト, 失敗時エラー要約) を取り出す。

    ghdag.llm.LLMResult（stdout / stderr / returncode / ok）を想定する。
    ok=False は retryable failure としてエラー要約を返す。
    """
    stdout = getattr(result, "stdout", None)
    ok = getattr(result, "ok", None)
    if stdout is None or ok is None:
        raise TypeError(
            f"call_llm must return LLMResult-like object, got {type(result).__name__}"
        )
    text = stdout if isinstance(stdout, str) else str(stdout)
    if ok:
        return text, None
    stderr = getattr(result, "stderr", "") or ""
    stderr_s = stderr if isinstance(stderr, str) else str(stderr)
    returncode = getattr(result, "returncode", None)
    summary = f"llm call failed (returncode={returncode}): {stderr_s[:200]}"
    return text, summary


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

    def _metadata(*, retry: bool) -> dict[str, Any]:
        return {
            "duration_ms": int((time.monotonic() - start) * 1000),
            "trace_id": trace_id,
            "token_usage": None,
            "token_usage_reason": "call_llm does not expose token usage",
            "retry": retry,
        }

    def _failure_trace(raw: str, error: str, *, retry: bool) -> LlmTrace:
        return LlmTrace(
            input=prompt,
            raw_output=raw,
            parsed=None,
            reasoning="",
            config=config,
            metadata=_metadata(retry=retry),
            uncertain_flag=False,
            error=error,
        )

    def _attempt(p: str) -> tuple[str, dict[str, Any] | None, str, str | None]:
        try:
            result = call_llm(p, engine=engine, model=model)
        except Exception as exc:
            raise LlmCallError(
                str(exc), _failure_trace("", str(exc), retry=p != prompt)
            ) from exc
        try:
            raw, call_err = _unwrap_llm_result(result)
        except TypeError as exc:
            return str(result), None, "", str(exc)
        if call_err is not None:
            return raw, None, "", call_err
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
        if err2 is not None:
            trace = _failure_trace(raw2, err2, retry=True)
            raise LlmCallError(err2, trace) from None
        result = validator(parsed2)  # type: ignore[arg-type]
        trace = LlmTrace(
            input=prompt,
            raw_output=raw2,
            parsed=parsed2,
            reasoning=reasoning2,
            config=config,
            metadata=_metadata(retry=True),
            uncertain_flag=bool(parsed2.get("uncertain_flag", False)) if parsed2 else False,
        )
        return result, trace

    result = validator(parsed)  # type: ignore[arg-type]
    trace = LlmTrace(
        input=prompt,
        raw_output=raw,
        parsed=parsed,
        reasoning=reasoning,
        config=config,
        metadata=_metadata(retry=False),
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


def run_replan(
    instruction: str,
    *,
    engine: str,
    model: str,
    existing_ids: set[str],
    required_keep: set[str],
    max_subtasks: int,
) -> tuple[ReplanResponse, LlmTrace]:
    return _call_with_retry(
        instruction,
        engine=engine,
        model=model,
        validator=lambda d: _validate_replan(
            d,
            existing_ids=existing_ids,
            required_keep=required_keep,
            max_subtasks=max_subtasks,
        ),
    )


def run_evaluate(
    instruction: str,
    *,
    engine: str,
    model: str,
) -> tuple[EvaluateResponse, LlmTrace]:
    return _call_with_retry(instruction, engine=engine, model=model, validator=_validate_evaluate)


def run_classify_comment(
    instruction: str,
    *,
    engine: str,
    model: str,
) -> tuple[CommentClassifyResponse, LlmTrace]:
    return _call_with_retry(
        instruction, engine=engine, model=model, validator=_validate_comment_classify
    )


def run_reply_comment(
    instruction: str,
    *,
    engine: str,
    model: str,
) -> tuple[CommentReplyResponse, LlmTrace]:
    return _call_with_retry(
        instruction, engine=engine, model=model, validator=_validate_comment_reply
    )


def build_clarify_instruction(
    body: str,
    *,
    round_num: int,
    max_rounds: int,
    clarification_context: list[str] | None = None,
) -> str:
    context = ""
    if clarification_context:
        context = "\nClarification history:\n" + "\n".join(clarification_context) + "\n"
    return (
        f"You are clarifying an objective (round {round_num}/{max_rounds}).\n"
        f"Objective:\n{body}\n{context}\n"
        "Respond with JSON: "
        '{"clear": bool, "question": str|null, "reason": str, '
        '"reasoning": str, "uncertain_flag": bool}'
    )


_DELIVERABLE_CONTRACT = (
    "Each auto subtask must edit the single shared deliverable.md in place. "
    "Do not create new draft or deliverable files."
)

_SUBTASK_SCHEMA = (
    '{"id": str, "title": str, "kind": "auto"|"human"|"watch", "prompt": str, '
    '"condition"?: object, "depends"?: [str], "timeout_sec"?: int, "poll_interval_sec"?: int}'
)


def build_decompose_instruction(
    body: str,
    *,
    iteration: int,
    max_subtasks: int,
    next_focus: str = "",
    clarification_context: list[str] | None = None,
) -> str:
    extra = f"\nFocus for this iteration: {next_focus}\n" if next_focus else ""
    if clarification_context:
        extra += "\nClarification history:\n" + "\n".join(clarification_context) + "\n"
    return (
        f"Decompose the objective into subtasks (iteration {iteration}). "
        f"Maximum {max_subtasks} subtasks.\n"
        f"{_DELIVERABLE_CONTRACT}\n"
        "kind=watch requires condition; auto/human require non-empty prompt.\n"
        "Omit depends for sequential order; use depends: [] for parallel-ready watch.\n"
        f"Objective:\n{body}{extra}\n\n"
        "Respond with JSON: "
        f'{{"subtasks": [{_SUBTASK_SCHEMA}], '
        '"reasoning": str, "uncertain_flag": bool}'
    )


def build_replan_instruction(
    body: str,
    *,
    plan_summary: str,
    failure_detail: str,
    deliverable_excerpt: str = "",
    human_feedback: str = "",
) -> str:
    feedback = f"\nHuman revision feedback:\n{human_feedback}\n" if human_feedback else ""
    deliverable_block = ""
    if deliverable_excerpt:
        deliverable_block = f"\n\nCurrent deliverable excerpt:\n{deliverable_excerpt}\n"
    return (
        "Replan after a watch failure or human plan revision.\n"
        f"Objective:\n{body}\n\n"
        f"Current plan:\n{plan_summary}\n\n"
        f"Failure / trigger detail:\n{failure_detail}\n"
        f"{feedback}{deliverable_block}\n"
        "Respond with JSON: "
        '{"keep": [str], "add": ['
        f"{_SUBTASK_SCHEMA}"
        '], "reason": str, "reasoning": str, "uncertain_flag": bool}. '
        "Always keep running and success task ids."
    )


def build_evaluate_instruction(
    body: str,
    *,
    results_summary: str,
    iteration: int,
    max_iterations: int,
    deliverable_excerpt: str = "",
) -> str:
    deliverable_block = ""
    if deliverable_excerpt:
        deliverable_block = f"\n\nCurrent deliverable excerpt:\n{deliverable_excerpt}\n"
    return (
        f"Evaluate progress (iteration {iteration}/{max_iterations}).\n"
        f"Objective:\n{body}\n\nResults:\n{results_summary}"
        f"{deliverable_block}\n"
        "Respond with JSON: "
        '{"achieved": bool, "score": int(0-100), "summary": str, "next_focus": str, '
        '"reasoning": str, "uncertain_flag": bool}'
    )


def build_auto_subtask_prompt(
    work_prompt: str,
    *,
    deliverable_path: str,
    deliverable_excerpt: str,
) -> str:
    """decompose 作業指示に deliverable 編集契約ブロックを付加する。"""
    return (
        f"{work_prompt}\n\n"
        "---\n"
        f"Deliverable path: {deliverable_path}\n"
        "Edit this file directly. Do not create new deliverable or draft files.\n"
        "Stdout must be a 3-5 line summary of changes only.\n\n"
        f"Current deliverable excerpt:\n{deliverable_excerpt}\n"
    )


def build_comment_classify_instruction(comment_text: str) -> str:
    return (
        "Classify the following user comment on an in-progress work loop.\n"
        f"Comment:\n{comment_text}\n\n"
        "Choose exactly one intent:\n"
        "- status: asking for progress / whether work is running\n"
        "- instruction: requesting a plan change or correction\n"
        "- question: asking a substantive question about the work\n"
        "- chitchat: acknowledgment or unrelated small talk\n\n"
        "Respond with JSON: "
        '{"intent": "status"|"instruction"|"question"|"chitchat", '
        '"reason": str, "reasoning": str, "uncertain_flag": bool}'
    )


def build_comment_reply_instruction(
    *,
    objective: str,
    deliverable_excerpt: str,
    plan_summary: str,
    recent_results: str,
    comment_text: str,
    max_chars: int,
) -> str:
    return (
        "Answer the user's question about an in-progress work loop.\n"
        f"Objective:\n{objective}\n\n"
        f"Deliverable excerpt:\n{deliverable_excerpt}\n\n"
        f"Current plan:\n{plan_summary}\n\n"
        f"Recent subtask results:\n{recent_results}\n\n"
        f"User question:\n{comment_text}\n\n"
        f"Keep the reply within {max_chars} characters.\n"
        "Respond with JSON: "
        '{"reply": str, "reasoning": str, "uncertain_flag": bool}'
    )


def validate_decompose_payload(
    data: dict[str, Any], *, max_subtasks: int
) -> DecomposeResponse:
    """テスト／呼び出し元向けの公開検証入口。"""
    return _validate_decompose(data, max_subtasks=max_subtasks)


def validate_replan_payload(
    data: dict[str, Any],
    *,
    existing_ids: set[str],
    required_keep: set[str],
    max_subtasks: int,
) -> ReplanResponse:
    """テスト／呼び出し元向けの公開検証入口。"""
    return _validate_replan(
        data,
        existing_ids=existing_ids,
        required_keep=required_keep,
        max_subtasks=max_subtasks,
    )


def validate_comment_classify_payload(data: dict[str, Any]) -> CommentClassifyResponse:
    return _validate_comment_classify(data)


def validate_comment_reply_payload(data: dict[str, Any]) -> CommentReplyResponse:
    return _validate_comment_reply(data)
