"""
mltgnt.skill — Markdown ベーススキルファイルの読み込み・実行基盤。

設計: Issue #124
公開 API: discover, load, match, run
"""
from mltgnt.skill._registry import SkillRegistry
from mltgnt.skill.loader import discover, load
from mltgnt.skill.matcher import match
from mltgnt.skill.models import SkillFile, SkillMeta
from mltgnt.skill.runner import run

__all__ = [
    "discover",
    "load",
    "match",
    "run",
    "SkillMeta",
    "SkillFile",
    "SkillRegistry",
]
