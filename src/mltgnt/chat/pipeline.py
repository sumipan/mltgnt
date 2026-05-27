"""mltgnt.chat.pipeline

1 往復パイプラインの本体。
ペルソナ読み込み → プロンプト整形 → LLM 呼び出し → 応答整形 → audit 記録。
"""
from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mltgnt.interfaces.types import ChatOutput

logger = logging.getLogger(__name__)


def run_pipeline(
    prompt: str,
    persona_name: str,
    persona_dir: Path,
    *,
    timeout: int = 300,
    memory: str | None = None,
    audit_writer: Callable[[dict], None] | None = None,
) -> ChatOutput:
    """1 往復パイプラインの本体。

    Returns:
        ChatOutput。エラー時は content にエラー文字列を格納。例外は送出しない。
    """
    from mltgnt.bridges.llm_adapter import call_llm
    from mltgnt.persona.loader import load
    from mltgnt.persona.registry import resolve_with_alias
    from mltgnt.persona.schema import SYSTEM_DEFAULT_ENGINE, SYSTEM_DEFAULT_MODEL

    path = resolve_with_alias(str(persona_name), persona_dir)
    persona = load(path)

    effective_prompt = f"{memory}\n\n{prompt}" if memory is not None else prompt
    formatted = persona.format_prompt(effective_prompt)
    engine = persona.fm.engine or SYSTEM_DEFAULT_ENGINE
    model = persona.fm.model or SYSTEM_DEFAULT_MODEL

    logger.debug("[pipeline] persona=%r engine=%r", persona_name, engine)

    ok = False
    content: str
    try:
        result = call_llm(formatted, engine=engine, model=model, timeout=timeout)
        ok = result.ok
    except Exception as e:
        logger.warning("[pipeline] persona=%r exception: %s", persona_name, e)
        content = f"（実行失敗: {e}）"
    else:
        if not result.ok:
            stderr = (result.stderr or "").strip()
            logger.warning("[pipeline] persona=%r ok=False stderr=%s", persona_name, stderr[:200])
            content = f"（エラー: {stderr[:200]}）" if stderr else "（エラー）"
        else:
            content = (result.stdout or "").strip()

    if audit_writer is not None:
        try:
            audit_writer({
                "source": f"mltgnt-persona-{persona_name}",
                "engine": engine,
                "model": model,
                "ok": ok,
                "timestamp": datetime.now(tz=ZoneInfo("Asia/Tokyo")).isoformat(),
            })
        except Exception:
            logger.warning("[pipeline] audit_writer failed for persona=%r", persona_name)

    return ChatOutput(
        content=content,
        persona_name=persona_name,
        timestamp=datetime.now(tz=ZoneInfo("Asia/Tokyo")),
        session_key="",
    )


def run_chat(
    prompt: str,
    persona_name: str,
    persona_dir: Path,
    *,
    timeout: int = 300,
    memory: str | None = None,
    audit_writer: Callable[[dict], None] | None = None,
) -> ChatOutput:
    """Deprecated: use run_pipeline instead."""
    warnings.warn(
        "run_chat() is deprecated, use run_pipeline()",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_pipeline(
        prompt,
        persona_name,
        persona_dir,
        timeout=timeout,
        memory=memory,
        audit_writer=audit_writer,
    )
