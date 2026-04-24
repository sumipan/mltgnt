"""llm_fallback 経路向けのキーワードベーススキル推定。"""

from __future__ import annotations

from pathlib import Path

from mltgnt.skill.loader import discover


def estimate_skill(
    instruction: str,
    skill_paths: list[str | Path],
) -> str | None:
    """instruction に含まれる triggers キーワードからスキル名を推定する。

    Args:
        instruction: ユーザー発話テキスト。
        skill_paths: SKILL.md を検索するディレクトリパスのリスト。

    Returns:
        マッチしたスキル名（例: "calendar"）。マッチなしは None。
    """
    skills = discover([Path(p) for p in skill_paths])
    for meta in skills.values():
        for trigger in meta.triggers:
            if trigger in instruction:
                return meta.name
    return None
