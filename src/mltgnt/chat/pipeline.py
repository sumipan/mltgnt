"""チャットパイプライン — memory 注入."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mltgnt.config import ChatConfig, MemoryConfig

_log = logging.getLogger(__name__)


def run_pipeline(
    config: "ChatConfig",
    memory_config: "MemoryConfig",
    user_query: str,
    persona_stem: str,
) -> str:
    """memory excerpt を取得して返す."""
    from mltgnt.memory import read_memory_with_sufficiency_check

    return read_memory_with_sufficiency_check(
        memory_config,
        persona_stem,
        user_query,
        max_bytes=memory_config.inject_max_bytes,
        max_entries=memory_config.inject_max_entries,
    )
