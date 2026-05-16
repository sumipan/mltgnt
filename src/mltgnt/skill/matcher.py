"""
mltgnt.skill.matcher — スキルマッチング（スラッシュ / triggers / LLM）。

設計: Issue #124 §6.3, Issue #208
"""
from __future__ import annotations

import re
import sys

from ghdag.llm import call as llm_call

from mltgnt.skill.models import SkillMeta

_SLASH_PATTERN = re.compile(r"^/(\S+)(.*)", re.DOTALL)

_DEFAULT_MATCHER_MODEL = "claude-haiku-4-5-20251001"

_LLM_SYSTEM_PROMPT = """\
あなたはスキルマッチャーです。
ユーザー入力が以下のスキル一覧のどれかに対応するか判定してください。
対応するスキルがあればそのスキル名のみを返してください。
どれにも対応しない場合は "none" とだけ返してください。
余計な説明は不要です。
"""


def _filter_by_persona(
    skills: dict[str, SkillMeta],
    persona_skills: list[str] | None,
) -> dict[str, SkillMeta]:
    if persona_skills is None:
        return skills
    return {k: v for k, v in skills.items() if k in persona_skills}


def _match_by_triggers(
    user_input: str,
    skills: dict[str, SkillMeta],
    persona_skills: list[str] | None,
) -> tuple[SkillMeta, str] | None:
    """triggers キーワードの部分一致でスキルを検索する。

    Args:
        user_input: ユーザー入力文字列
        skills: discover() が返すスキル辞書
        persona_skills: ペルソナに許可されたスキル名リスト（None なら制限なし）

    Returns:
        (SkillMeta, user_input) または None
        triggers マッチ時は user_input 全文を arguments として渡す
    """
    filtered = _filter_by_persona(skills, persona_skills)
    for meta in filtered.values():
        for trigger in meta.triggers:
            if trigger in user_input:
                return (meta, user_input)
    return None


async def _match_by_llm(
    user_input: str,
    skills: dict[str, SkillMeta],
    persona_skills: list[str] | None,
    model: str | None = None,
) -> tuple[SkillMeta, str] | None:
    """LLM にスキル一覧と入力を渡し、意図分類する。

    Args:
        user_input: ユーザー入力文字列
        skills: discover() が返すスキル辞書
        persona_skills: ペルソナに許可されたスキル名リスト（None なら制限なし）

    Returns:
        (SkillMeta, user_input) または None
        LLM が「none」を返した場合、または応答が登録スキル名と一致しない場合は None
    """
    filtered = _filter_by_persona(skills, persona_skills)
    if not filtered:
        return None

    skill_list = "\n".join(
        f"- {meta.name}: {meta.description}" for meta in filtered.values()
    )
    prompt = f"{_LLM_SYSTEM_PROMPT}\n\nスキル一覧:\n{skill_list}\n\nユーザー入力: {user_input}"

    try:
        result = llm_call(prompt, engine="claude", model=model or _DEFAULT_MATCHER_MODEL, timeout=30)
        if not result.ok:
            print(f"[skill.matcher] LLM 意図分類エラー: {result.stderr}", file=sys.stderr)
            return None
        response = result.stdout.strip().lower()
    except Exception as e:
        print(f"[skill.matcher] LLM 意図分類エラー: {e}", file=sys.stderr)
        return None

    if response == "none" or response not in filtered:
        return None

    return (filtered[response], user_input)


async def match(
    user_input: str,
    skills: dict[str, SkillMeta],
    persona_skills: list[str] | None = None,
    model: str | None = None,
) -> tuple[SkillMeta, str] | None:
    """
    ユーザー入力からスキルを特定する（ハイブリッド 2段フォールバック）。

    優先順位: スラッシュコマンド → triggers 部分一致 → LLM 意図分類

    戻り値: (SkillMeta, arguments_str) のタプル。マッチしなければ None。
    """
    # Step 1: スラッシュコマンド
    m = _SLASH_PATTERN.match(user_input)
    if m:
        name = m.group(1)
        rest = m.group(2)
        arguments = rest.lstrip(" ") if rest else ""
        if name not in skills:
            return None
        meta = skills[name]
        if persona_skills is not None and name not in persona_skills:
            return None
        return (meta, arguments)

    # Step 2: triggers 部分一致
    result = _match_by_triggers(user_input, skills, persona_skills)
    if result is not None:
        return result

    # Step 3: LLM 意図分類
    return await _match_by_llm(user_input, skills, persona_skills, model=model)
