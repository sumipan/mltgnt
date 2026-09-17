"""
mltgnt.skill — load and run Markdown-based skill files.

Design: Issue #124
Public API: discover, load, match, run, estimate_skill, resolve_skill
"""
from __future__ import annotations

from pathlib import Path

from mltgnt.skill._registry import SkillRegistry
from mltgnt.skill.context import build_extra_context
from mltgnt.skill.lint import lint_skill_meta
from mltgnt.skill.loader import discover, load
from mltgnt.skill.matcher import match
from mltgnt.skill.models import (
    ArtifactSpec,
    ConsumesSpec,
    ExitStatus,
    ProducesSpec,
    SkillFile,
    SkillMatchResult,
    SkillMeta,
    SkillRunResult,
)
from mltgnt.skill.runner import run

__all__ = [
    "discover",
    "discover_bodies",
    "load",
    "match",
    "resolve_skill",
    "run",
    "build_extra_context",
    "SkillMeta",
    "SkillFile",
    "SkillRegistry",
    "ArtifactSpec",
    "ProducesSpec",
    "ConsumesSpec",
    "SkillRunResult",
    "SkillMatchResult",
    "lint_skill_meta",
]


def discover_bodies(paths: list[Path]) -> list[str]:
    """discover + load body. Convenience for passing skill text to the memory layer."""
    from mltgnt.bridges.files_adapter import md_read

    path_list = [Path(p) for p in paths]
    skills = discover(path_list)
    return [
        md_read(meta.path.name, repo_root=meta.path.parent).content
        for meta in skills.values()
    ]


async def resolve_skill(
    user_input: str,
    skill_paths: list,
    persona_skills: list[str] | None = None,
    entry_file: str = "SKILL.md",
    matcher_model: str | None = None,
) -> "tuple | None":
    """
    Search/match a skill from user input; return (SkillFile, arguments_str).

    Return None (not an error) when skill_paths is empty or missing.
    Also return None when no skill matches.

    Args:
        user_input: User message string
        skill_paths: List of skill directories (Path or str)
        persona_skills: Persona skills field (None = no filter)
        entry_file: Skill entry filename
        matcher_model: Model for LLM intent classification (None = default)
    Returns:
        (SkillFile, arguments_str) or None
    """
    from pathlib import Path as _Path

    paths = [_Path(p) for p in skill_paths]
    skills = discover(paths, entry_file=entry_file)
    if not skills:
        return None

    result = await match(user_input, skills, persona_skills=persona_skills, model=matcher_model)
    if result.decisive is None:
        return None

    skill_file = load(result.decisive)
    return (skill_file, result.arguments)
