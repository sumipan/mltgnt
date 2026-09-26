"""Opaque conversation id <-> (space, thread) mapping.

Only media implementations split a conversation id; everything inward treats it
as an opaque string.
"""

from __future__ import annotations

__all__ = ["resolve", "storage_key", "storage_key_from_conversation_id", "to_conversation_id"]

_SEP = ":"


def to_conversation_id(space: str, thread: str) -> str:
    """Build an opaque conversation id from a space and a thread id."""
    return f"{space}{_SEP}{thread}"


def resolve(conversation_id: str) -> tuple[str, str]:
    """Split a conversation id back into ``(space, thread)``. Raise ValueError if malformed."""
    space, sep, thread = conversation_id.partition(_SEP)
    if not sep or not space or not thread:
        raise ValueError(f"invalid conversation_id: {conversation_id!r}")
    return space, thread


def storage_key(space: str, thread: str) -> str:
    """Directory-safe key for per-thread state (``<space>-<thread>``)."""
    return f"{space}-{thread}"


def storage_key_from_conversation_id(conversation_id: str) -> str:
    return storage_key(*resolve(conversation_id))
