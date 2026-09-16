"""mltgnt.persona.types — ペルソナ層の境界データ（#3318）。

媒体固有の語（channel / thread_ts 等）を持たない。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaContext:
    """判断層へ渡すペルソナ材料。"""

    persona_id: str
    engine: str | None = None
    model: str | None = None
    profile: str = ""
    memory_excerpt: str = ""
    skills: tuple[str, ...] = ()
    phrases: tuple[str, ...] = ()
    observers: tuple[str, ...] = ()
