"""mltgnt.persona — ペルソナの型定義とバリデーション。"""

from __future__ import annotations

from dataclasses import dataclass

from mltgnt.persona.schema import PersonaFM, ValidationResult, validate_fm, validate_sections

__all__ = [
    "Persona",
    "PersonaFM",
    "ValidationResult",
    "validate_fm",
    "validate_sections",
    "validate_persona",
]


@dataclass
class Persona:
    """ペルソナ（フロントマター + 本文）。"""

    name: str
    fm: PersonaFM
    body: str


def validate_persona(
    persona: Persona,
    *,
    available_skills: list[str] | None = None,
    required_sections: tuple[str, ...] | None = None,
) -> list[str]:
    """
    ペルソナ全体のバリデーション。

    Returns:
        警告メッセージのリスト
    """
    messages: list[str] = []

    # 1. FM バリデーション
    fm_result = validate_fm(persona.fm)
    messages.extend(fm_result.warnings)
    messages.extend(fm_result.errors)

    # name ミスマッチ
    if persona.name != persona.fm.name:
        messages.append(
            f"ペルソナ名が一致しません: ファイル名「{persona.name}」、FM「{persona.fm.name}」"
        )

    # 2. スキル検証
    if available_skills is not None:
        for skill in persona.fm.skills:
            if skill not in available_skills:
                messages.append(f"スキル「{skill}」は利用可能なスキルにありません")

    # 3. セクション検証
    section_result = validate_sections(persona.body, persona.fm, required_sections)
    messages.extend(section_result.warnings)
    messages.extend(section_result.errors)

    return messages
