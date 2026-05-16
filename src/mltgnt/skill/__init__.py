"""
mltgnt.skill — Markdown ベーススキルファイルの読み込み・実行基盤。

設計: Issue #124
公開 API: discover, load, match, run, estimate_skill, resolve_skill
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
    "resolve_skill",
    "run",
    "SkillMeta",
    "SkillFile",
    "SkillRegistry",
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
    if result is None:
        return None

    meta, arguments = result
    skill_file = load(meta)
    return (skill_file, arguments)
