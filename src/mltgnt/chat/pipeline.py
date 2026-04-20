"""チャットパイプライン — memory 注入と十分性判定."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mltgnt.config import ChatConfig, MemoryConfig
    from mltgnt.memory._sufficiency import LlmCall

_log = logging.getLogger(__name__)


def _build_sufficiency_llm_call(engine: str, model: str | None = None) -> "LlmCall":
    """十分性判定用の LLM 呼び出し関数を生成する (ghdag.llm.call ベース)."""
    from ghdag.llm import call as ghdag_llm_call

    def llm_call(prompt: str) -> str:
        result = ghdag_llm_call(prompt, engine=engine, model=model, timeout=30)
        if not result.ok:
            raise RuntimeError(f"LLM call failed: {result.stderr}")
        return result.stdout.strip()

    return llm_call


def run_pipeline(
    config: "ChatConfig",
    memory_config: "MemoryConfig",
    user_query: str,
    persona_stem: str,
) -> str:
    """memory excerpt を取得して返す.

    sufficiency_engine が設定されている場合は十分性判定付き検索を行う。
    未設定の場合は read_memory_by_relevance() と同じ動作（後方互換）。
    """
    from mltgnt.memory import read_memory_with_sufficiency_check

    llm_call = None
    if config.sufficiency_engine is not None:
        llm_call = _build_sufficiency_llm_call(
            config.sufficiency_engine,
            config.sufficiency_model,
        )

    return read_memory_with_sufficiency_check(
        memory_config,
        persona_stem,
        user_query,
        max_bytes=memory_config.inject_max_bytes,
        max_entries=memory_config.inject_max_entries,
        llm_call=llm_call,
    )
