"""src/mltgnt/routing/channel_router.py

Multi-space (media-agnostic) agent routing logic.

Functions that decide who should respond to a message.
Priority: 1. nickname 2. conversation pin 3. primary 4. None

Design: Issue #284 / #3285
"""
from __future__ import annotations

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
