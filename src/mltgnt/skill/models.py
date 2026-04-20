"""
mltgnt.skill.models — SkillMeta / SkillFile dataclass 定義。

設計: Issue #124 §6.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillMeta:
    """discover 時にロードされるメタ情報（Progressive Disclosure）。"""

    name: str
    description: str
    argument_hint: str
    model: str | None
    path: Path
    triggers: list[str] = field(default_factory=list)


@dataclass
class SkillFile:
    """実行時にロードされる全文データ。"""

    meta: SkillMeta
    body: str
