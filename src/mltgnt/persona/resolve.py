"""mltgnt.persona.resolve — choose responder and PersonaContext (#3318).

Routing / host-specific loaders are injected by the caller (layer boundary, media-agnostic).
"""
from __future__ import annotations

from typing import Any, Callable

from mltgnt.persona.types import PersonaContext

ResolveFn = Callable[..., str | None]
LoadProfileFn = Callable[..., tuple[str | None, str | None]]
EngineModelFn = Callable[[str], tuple[str, str] | None]
SkillsFn = Callable[[str], list[str] | None]
PhrasesFn = Callable[[str], dict[str, str]]


def resolve_responder(
    text: str,
    *,
    space_id: str,
    conversation_id: str | None,
    persona_map: dict[str, list[Any]],
    pinned_personas: dict[str, str],
    resolve_fn: ResolveFn,
) -> str | None:
    """Return the responder persona name. None if no reply is needed.

    space_id / conversation_id are opaque strings (media mapping is the caller's job).
    """
    return resolve_fn(
        text,
        space_id=space_id,
        conversation_id=conversation_id,
        persona_map=persona_map,
        pinned_personas=pinned_personas,
    )


def build_persona_context(
    persona_id: str,
    *,
    weight: str = "heavy",
    memory_excerpt: str = "",
    observers: tuple[str, ...] = (),
    load_profile_fn: LoadProfileFn,
    engine_model_fn: EngineModelFn,
    skills_fn: SkillsFn,
    phrases_fn: PhrasesFn,
) -> PersonaContext:
    """Build PersonaContext for the decision layer (loader is host-injected)."""
    profile, _err = load_profile_fn(persona_id, weight=weight)
    em = engine_model_fn(persona_id)
    skills = skills_fn(persona_id) or []
    phrases_map = phrases_fn(persona_id)
    return PersonaContext(
        persona_id=persona_id,
        engine=em[0] if em else None,
        model=em[1] if em else None,
        profile=profile or "",
        memory_excerpt=memory_excerpt or "",
        skills=tuple(skills),
        phrases=tuple(phrases_map.values()),
        observers=observers,
    )
