"""mltgnt.persona — ペルソナ管理モジュール

公開 API:
    load_persona(name, *, persona_dir)                        -> Persona
    list_personas(persona_dir)                                -> list[str]
    validate_persona(persona, *, available_skills)            -> list[str]
    run_persona_prompt(persona_name, prompt, persona_dir, ..) -> str
    PersonaValidationError                                    (例外クラス)
"""

from __future__ import annotations

from pathlib import Path

from mltgnt.persona.loader import Persona, load
from mltgnt.persona.registry import list_personas as _list_personas
from mltgnt.persona.registry import resolve_with_alias
from mltgnt.persona.runner import run_persona_prompt

__all__ = [
    "Persona",
    "PersonaValidationError",
    "load_persona",
    "list_personas",
    "validate_persona",
    "run_persona_prompt",
]


# ---------------------------------------------------------------------------
# 例外クラス
# ---------------------------------------------------------------------------


class PersonaValidationError(Exception):
    """ペルソナ定義が不正な場合に送出される例外。"""
    pass


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def load_persona(name: str, *, persona_dir: Path | None = None) -> Persona:
    """名前またはエイリアスでペルソナを読み込む。

    Args:
        name: ペルソナ名またはエイリアス
        persona_dir: ペルソナファイルのディレクトリ（デフォルト: ./agents/）

    Returns:
        Persona dataclass

    Raises:
        FileNotFoundError: 該当ペルソナが見つからない
        PersonaValidationError: frontmatter が不正
    """
    pdir = persona_dir if persona_dir is not None else Path("agents")
    path = resolve_with_alias(name, pdir)
    return load(path)


def list_personas(persona_dir: Path | None = None) -> list[str]:
    """利用可能なペルソナ名一覧を返す。

    Args:
        persona_dir: ペルソナファイルのディレクトリ（デフォルト: ./agents/）

    Returns:
        ペルソナ名（stem）のリスト（名前順）
    """
    pdir = persona_dir if persona_dir is not None else Path("agents")
    return _list_personas(pdir)


def validate_persona(
    persona: Persona,
    *,
    available_skills: list[str] | None = None,
) -> list[str]:
    """ペルソナ定義を検証し、警告メッセージのリストを返す。空リストなら正常。

    Args:
        persona: 検証対象の Persona オブジェクト
        available_skills: 利用可能なスキル名リスト。
                          None の場合はスキルチェックをスキップする。

    Returns:
        警告メッセージのリスト（空リストなら正常）
    """
    messages: list[str] = []

    # persona.name とファイル名(stem)の一致チェック
    if persona.fm.name and persona.path.stem != persona.fm.name:
        messages.append(
            f"persona.name ({persona.fm.name!r}) がファイル名 ({persona.path.stem!r}) と不一致です"
        )

    # 未知 FM キーの警告
    for k in persona.fm.unknown_keys:
        messages.append(f"未定義の FM キー: {k!r}")

    # 旧形式キーの警告
    if persona.fm.legacy_keys:
        messages.append(
            f"旧形式の FM キー {persona.fm.legacy_keys} が使用されています"
        )

    # スキルチェック（available_skills が指定された場合のみ）
    if available_skills is not None:
        available_set = set(available_skills)
        for skill in persona.fm.skills:
            if skill not in available_set:
                messages.append(f"未定義のスキル: {skill!r}")

    return messages
