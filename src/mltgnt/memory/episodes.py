"""mltgnt.memory.episodes — append user/assistant turn pairs to episodic memory."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mltgnt.memory.api import append_memory_entry, persona_memory_lock
from mltgnt.persona.memory import (
    LEGACY_DIRECT_SOURCE_TAG,
    direct_dedupe_base,
    format_tool_trace_block,
)

if TYPE_CHECKING:
    from mltgnt.config import MemoryConfig

__all__ = ["append_turn_episodes"]

_log = logging.getLogger(__name__)


def append_turn_episodes(
    config: "MemoryConfig",
    persona_stem: str,
    *,
    user_text: str,
    assistant_text: str,
    space_id: str,
    thread_key: str,
    message_id: str,
    timestamp: str,
    source_tag: str = LEGACY_DIRECT_SOURCE_TAG,
    tool_trace: list[dict] | None = None,
    lock_timeout_sec: float | None = None,
) -> bool:
    """Append a user and assistant line for one direct-chat turn (idempotent dedupe keys)."""
    dedupe_base = direct_dedupe_base(space_id, thread_key, message_id)
    user_key = f"{dedupe_base}:user"
    assistant_key = f"{dedupe_base}:assistant"
    if tool_trace:
        assistant_body = f"{format_tool_trace_block(tool_trace)}\n\n{assistant_text}"
    else:
        assistant_body = assistant_text

    with persona_memory_lock(config, persona_stem, timeout_sec=lock_timeout_sec) as ok:
        if not ok:
            _log.warning("append_turn_episodes: lock timeout persona=%s", persona_stem)
            return False
        user_ok = append_memory_entry(
            config,
            persona_stem,
            "user",
            user_text,
            timestamp,
            source_tag=source_tag,
            dedupe_key=user_key,
            under_lock=True,
        )
        assistant_ok = append_memory_entry(
            config,
            persona_stem,
            "assistant",
            assistant_body,
            timestamp,
            source_tag=source_tag,
            dedupe_key=assistant_key,
            under_lock=True,
        )
        if not (user_ok and assistant_ok):
            _log.warning(
                "append_turn_episodes: append failed persona=%s user_ok=%s assistant_ok=%s",
                persona_stem,
                user_ok,
                assistant_ok,
            )
            return False
    return True
