"""mltgnt.persona.result_format — ペルソナ整形の LLM 呼び出し骨格（#3318）。

プロンプト本文・LLM 呼び出し・媒体向け後処理はホスト注入。
媒体固有の Markdown 変換はホスト側に残す。
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
    """LLM で結果テキストをペルソナスタイルに整形する。失敗時は None。

    Args:
        raw_body: 整形対象の下書き本文
        prompt_header: ホストが組み立てたプロンプト先頭（人物像・方針など）
        llm_call: ``(prompt, *, stdin_text, engine, model, timeout) -> result``
        logger: ``warning`` を持つロガー
        postprocess: 媒体向け後処理（未指定時は strip のみ）
    """
    s = (raw_body or "").strip()
    if not s:
        return None
    if len(s) > input_max_chars:
        s = s[:input_max_chars] + "\n\n[入力が長いため途中まで。以降は省略されている]"
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
