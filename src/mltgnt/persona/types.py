"""mltgnt.persona.types — persona-layer boundary data (#3318).

No media-specific terms (channel / thread_ts, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaContext:
    """Persona materials passed to the decision layer."""

    persona_id: str
    engine: str | None = None
    model: str | None = None
    profile: str = ""
    memory_excerpt: str = ""
    skills: tuple[str, ...] = ()
    phrases: tuple[str, ...] = ()
    observers: tuple[str, ...] = ()
