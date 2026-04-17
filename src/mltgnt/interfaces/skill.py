"""
mltgnt.interfaces.skill — SkillLoaderProtocol 定義。

設計: Issue #124 §3
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mltgnt.skill.models import SkillFile, SkillMeta


class SkillLoaderProtocol(Protocol):
    def discover(
        self,
        paths: list[Path],
        entry_file: str = "SKILL.md",
    ) -> dict[str, SkillMeta]:
        """SKILL.md を探索してメタ情報の dict を返す。"""
        ...

    def load(self, meta: SkillMeta) -> SkillFile:
        """SkillMeta から全文データをロードする。"""
        ...
