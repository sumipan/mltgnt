"""src/mltgnt/routing/channel_router.py

Multi-space (media-agnostic) agent routing logic.

Functions that decide who should respond to a message.
Priority: 1. nickname 2. conversation pin 3. primary 4. None

Design: Issue #284 / #3285
"""
from __future__ import annotations

import warnings

from mltgnt.routing import SpacePersonaEntry


def detect_nickname(
    text: str,
    entries: list[SpacePersonaEntry],
) -> str | None:
    """Return a persona name whose nickname appears in text. First match wins."""
    for entry in entries:
        if entry.nickname and entry.nickname in text:
            return entry.name
    return None


def resolve_persona(
    text: str,
    *,
    space_id: str,
    conversation_id: str | None,
    persona_map: dict[str, list[SpacePersonaEntry]],
    pinned_personas: dict[str, str],
) -> str | None:
    """Return the persona who should reply. None if no reply is needed.

    Priority: nickname → conversation pin (pinned_personas) → primary → None.
    space_id / conversation_id are opaque (media mapping is the caller's job).
    """
    entries = persona_map.get(space_id)
    if not entries:
        return None

    # 1. Nickname detection (before conversation pin)
    nickname_persona = detect_nickname(text, entries)
    if nickname_persona is not None:
        return nickname_persona

    # 2. Conversation pin (ignore personas not in the space)
    if conversation_id is not None:
        fixed = pinned_personas.get(conversation_id)
        if fixed is not None and any(e.name == fixed for e in entries):
            return fixed

    # 3. Primary persona
    for entry in entries:
        if entry.role == "primary":
            return entry.name

    # 4. None
    return None


def resolve_responding_persona(
    channel: str,
    text: str,
    thread_ts: str | None,
    channel_map: dict[str, list[SpacePersonaEntry]],
    thread_persona_map: dict[str, str],
) -> str | None:
    """Compat wrapper. Calls resolve_persona.

    .. deprecated::
        Use :func:`resolve_persona` with ``space_id`` / ``conversation_id`` instead.
    """
    warnings.warn(
        "resolve_responding_persona is deprecated; use resolve_persona "
        "(space_id / conversation_id)",
        DeprecationWarning,
        stacklevel=2,
    )
    conversation_id = f"{channel}:{thread_ts}" if thread_ts else None
    return resolve_persona(
        text,
        space_id=channel,
        conversation_id=conversation_id,
        persona_map=channel_map,
        pinned_personas=thread_persona_map,
    )


def find_observers_in_space(
    space_id: str,
    responding_persona: str | None,
    persona_map: dict[str, list[SpacePersonaEntry]],
) -> list[str]:
    """Return persona names in the space that are not the responder."""
    entries = persona_map.get(space_id, [])
    observers: list[str] = []
    for entry in entries:
        if entry.name != responding_persona:
            observers.append(entry.name)
    return observers


def find_observers(
    channel: str,
    responding_persona: str | None,
    channel_map: dict[str, list[SpacePersonaEntry]],
) -> list[str]:
    """Compat wrapper. Calls find_observers_in_space.

    .. deprecated::
        Use :func:`find_observers_in_space` instead.
    """
    warnings.warn(
        "find_observers is deprecated; use find_observers_in_space",
        DeprecationWarning,
        stacklevel=2,
    )
    return find_observers_in_space(channel, responding_persona, channel_map)
