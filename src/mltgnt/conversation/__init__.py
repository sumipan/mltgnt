"""mltgnt.conversation — 媒体非依存の会話層（#3317）。

待機列・セッション台帳・スレッド固定・圧縮。くびれ型は mltgnt.interfaces.turn。
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
    """会話層の保存先・閾値を注入する。"""
    global _active_config
    _active_config = config
    thread_queue.configure(config)
    session_store.configure(config)
    thread_index.configure(config)
    thread_persona_store.configure(config)
    session_compact.configure(config)


def get_config() -> ConversationConfig:
    """現在の ConversationConfig を返す。未設定なら RuntimeError。"""
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
