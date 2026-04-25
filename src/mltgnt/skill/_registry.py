"""
mltgnt.skill._registry — スレッドセーフなスキルレジストリ。
"""
from __future__ import annotations
import logging
import threading
from pathlib import Path
from mltgnt.skill.loader import discover
from mltgnt.skill.models import SkillMeta

logger = logging.getLogger("mltgnt.skill.registry")

class SkillRegistry:
    def __init__(self, paths: list[Path], entry_file: str = "SKILL.md") -> None:
        self._paths = [Path(p) for p in paths]
        self._entry_file = entry_file
        self._lock = threading.RLock()
        self._skills: dict[str, SkillMeta] = {}

    def reload(self) -> dict[str, SkillMeta]:
        new_skills = discover(self._paths, entry_file=self._entry_file)
        with self._lock:
            self._skills = new_skills
        logger.info("SkillRegistry: %d スキルをロード", len(new_skills))
        return dict(new_skills)

    def get(self) -> dict[str, SkillMeta]:
        with self._lock:
            return dict(self._skills)
