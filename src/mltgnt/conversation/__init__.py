"""mltgnt.conversation — media-agnostic conversation layer (#3317).

Queue, session ledger, thread binding, compaction. Neck types live in mltgnt.interfaces.turn.
"""

from __future__ import annotations

from mltgnt.config import ConversationConfig

from . import (
    fake_media,
    session_compact,
    session_store,
    thread_index,
    thread_persona_store,
    thread_queue,
)
from .types import (
    Attachment,
    HistoryMessage,
    TurnInput,
    TurnResult,
)

_active_config: ConversationConfig | None = None


def configure(config: ConversationConfig) -> None:
    """Inject conversation-layer storage paths and thresholds."""
    global _active_config
    _active_config = config
    thread_queue.configure(config)
    session_store.configure(config)
    thread_index.configure(config)
    thread_persona_store.configure(config)
    session_compact.configure(config)


def get_config() -> ConversationConfig:
    """Return the active ConversationConfig. Raises RuntimeError if unset."""
    if _active_config is None:
        raise RuntimeError("mltgnt.conversation is not configured; call configure() first")
    return _active_config


__all__ = [
    "Attachment",
    "ConversationConfig",
    "HistoryMessage",
    "TurnInput",
    "TurnResult",
    "configure",
    "fake_media",
    "get_config",
    "session_compact",
    "session_store",
    "thread_index",
    "thread_persona_store",
    "thread_queue",
]
