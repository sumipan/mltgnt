"""
mltgnt.skill — Markdown ベーススキルファイルの読み込み・実行基盤。

設計: Issue #124
公開 API: discover, load, match, run, estimate_skill, resolve_skill
"""
from __future__ import annotations

from pathlib import Path

from mltgnt.skill._registry import SkillRegistry
from mltgnt.skill.lint import lint_skill_meta
from mltgnt.skill.loader import discover, load
from mltgnt.skill.matcher import match
from mltgnt.skill.models import (
    ArtifactSpec,
    ConsumesSpec,
    ProducesSpec,
    RunOutput,
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
    "SkillMeta",
    "SkillFile",
    "SkillRegistry",
    "ArtifactSpec",
    "ProducesSpec",
    "ConsumesSpec",
    "SkillRunResult",
    "SkillMatchResult",
    "RunOutput",
    "lint_skill_meta",
]


def discover_bodies(paths: list[Path]) -> list[str]:
    """discover + 本文読み込み。memory 層にスキル本文を渡すための便利関数。"""
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
    ユーザー入力からスキルを検索・マッチし、(SkillFile, arguments_str) を返す。

    skill_paths が空または存在しない場合は None を返す（エラーにしない）。
    スキルがマッチしない場合も None を返す。

    引数:
        user_input: ユーザーのメッセージ文字列
        skill_paths: スキルディレクトリのリスト（Path または str）
        persona_skills: ペルソナの skills フィールド（None = フィルタなし）
        entry_file: スキルエントリファイル名
        matcher_model: LLM 意図分類に使うモデル（None = デフォルト）
    戻り値:
        (SkillFile, arguments_str) または None
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
