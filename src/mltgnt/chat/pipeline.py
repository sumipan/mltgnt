"""mltgnt.chat.pipeline

1 往復パイプラインの本体。
ペルソナ読み込み → プロンプト整形 → LLM 呼び出し → 応答整形 → audit 記録。
"""
from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

from pathlib import Path

from mltgnt.bridges.audit_adapter import OrchestrationContext
from mltgnt.interfaces.persona import PersonaProtocol
from mltgnt.interfaces.types import ChatOutput

logger = logging.getLogger(__name__)


def run_pipeline(
    prompt: str,
    persona: PersonaProtocol,
    *,
    engine: str = "",
    model: str = "",
    timeout: int = 300,
    memory: str | None = None,
    orchestration_ctx: OrchestrationContext | None = None,
    audit_path: Path | None = None,
    audit_writer: Callable[[dict], None] | None = None,
) -> ChatOutput:
    """1 往復パイプラインの本体。

    Returns:
        ChatOutput。エラー時は content にエラー文字列を格納。例外は送出しない。
    """
    from mltgnt.bridges.llm_adapter import call_llm

    persona_name = persona.name
    effective_prompt = f"{memory}\n\n{prompt}" if memory is not None else prompt
    formatted = persona.format_prompt(effective_prompt)

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

    if orchestration_ctx is not None and audit_path is not None:
        try:
            orchestration_ctx.record_persona_call(
                audit_path,
                engine=engine,
                model=model,
                ok=ok,
            )
        except Exception:
            logger.warning("[pipeline] orchestration audit failed for persona=%r", persona_name)
    elif audit_writer is not None:
        warnings.warn(
            "audit_writer is deprecated; use orchestration_ctx + audit_path instead",
            DeprecationWarning,
            stacklevel=2,
        )
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
