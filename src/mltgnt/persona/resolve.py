"""mltgnt.persona.resolve — 応答者決定・PersonaContext（#3318）。

routing / ホスト固有ローダは呼び出し側が注入する（層境界・媒体非依存）。
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
    """応答者ペルソナ名を返す。応答不要なら None。

    space_id / conversation_id は不透明文字列（媒体固有の写像は呼び出し側）。
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
    """判断層へ渡す PersonaContext を組み立てる（ローダはホスト注入）。"""
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
