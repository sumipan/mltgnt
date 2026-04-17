"""
mltgnt.skill.matcher — 明示的 /name パターンでスキルを特定する。

設計: Issue #124 §6.3
"""
from __future__ import annotations

import re

from mltgnt.skill.models import SkillMeta

_SLASH_PATTERN = re.compile(r"^/(\S+)(.*)", re.DOTALL)


def match(
    user_input: str,
    skills: dict[str, SkillMeta],
    persona_skills: list[str] | None = None,
) -> tuple[SkillMeta, str] | None:
    """
    ユーザー入力から明示的 /name パターンでスキルを特定する。

    戻り値: (SkillMeta, arguments_str) のタプル。マッチしなければ None。
    arguments_str は /name 直後のスペースを除いた残りの文字列。
    """
    m = _SLASH_PATTERN.match(user_input)
    if not m:
        return None

    name = m.group(1)
    rest = m.group(2)
    # Strip leading single space (separator between /name and args)
    # Strip leading spaces (separator between /name and args).
    # AC-3-6: "/review  a  b  c" → "a  b  c" (all leading spaces stripped)
    arguments = rest.lstrip(" ") if rest else ""

    if name not in skills:
        return None

    meta = skills[name]

    if persona_skills is not None and name not in persona_skills:
        return None

    return (meta, arguments)
