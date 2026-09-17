"""mltgnt.persona.result_format — LLM call skeleton for persona formatting (#3318).

Prompt body, LLM call, and media post-processing are host-injected.
Media-specific Markdown conversion stays on the host.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol

from mltgnt.persona.schema import SYSTEM_DEFAULT_ENGINE

DEFAULT_FORMAT_INPUT_MAX_CHARS = 48_000
DEFAULT_FORMAT_TIMEOUT_SEC = 25


class _LLMResultLike(Protocol):
    body: str
    success: bool
    returncode: int
    stderr: str


LLMCall = Callable[..., _LLMResultLike]
PostprocessFn = Callable[[str], str]
LoggerLike = Any


def format_result_for_persona(
    raw_body: str,
    *,
    prompt_header: str,
    llm_call: LLMCall,
    logger: LoggerLike,
    engine: str = "",
    model: str = "",
    postprocess: PostprocessFn | None = None,
    input_max_chars: int = DEFAULT_FORMAT_INPUT_MAX_CHARS,
    timeout: int | None = DEFAULT_FORMAT_TIMEOUT_SEC,
) -> str | None:
    """Format result text into persona style via LLM. None on failure.

    Args:
        raw_body: Draft body to format
        prompt_header: Host-built prompt prefix (persona, policy, etc.)
        llm_call: ``(prompt, *, stdin_text, engine, model, timeout) -> result``
        logger: Logger with ``warning``
        postprocess: Media post-process (strip only when omitted)
    """
    s = (raw_body or "").strip()
    if not s:
        return None
    if len(s) > input_max_chars:
        s = s[:input_max_chars] + "\n\n[Input truncated because it was too long]"
    full_input = (prompt_header or "") + s

    effective_engine = (engine or "").strip() or SYSTEM_DEFAULT_ENGINE
    try:
        proc = llm_call(
            "",
            stdin_text=full_input,
            engine=effective_engine,
            model=model or None,
            timeout=timeout,
        )
    except Exception as e:
        logger.warning("result format: %s error: %s", effective_engine, e)
        return None

    if not getattr(proc, "success", False):
        err = (getattr(proc, "stderr", None) or "").strip()
        logger.warning(
            "result format: %s exit %s stderr=%s",
            effective_engine,
            getattr(proc, "returncode", None),
            err[:500],
        )
        return None

    out = (getattr(proc, "body", None) or "").strip()
    if postprocess is not None:
        formatted = postprocess(out)
    else:
        formatted = out
    if not formatted:
        return None
    return formatted
