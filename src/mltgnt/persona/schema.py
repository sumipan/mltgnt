"""persona.schema — PersonaFM / ValidationResult とバリデーション関数。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS: tuple[str, ...] = (
    "基本情報",
    "価値観",
    "反応パターン",
    "口調",
    "アウトプット形式",
)

# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class PersonaFM:
    """フロントマター（YAML ヘッダ）の構造化表現。"""

    name: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    skills: list[str] = field(default_factory=list)
    unknown_keys: list[str] = field(default_factory=list)
    legacy_keys: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """バリデーション結果。"""

    ok: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# バリデーション関数
# ---------------------------------------------------------------------------


def validate_fm(fm: PersonaFM) -> ValidationResult:
    """PersonaFM 単体のバリデーション。"""
    warnings: list[str] = []
    errors: list[str] = []

    if not fm.name:
        errors.append("name が空です")

    for key in fm.unknown_keys:
        warnings.append(f"未知のキー「{key}」が含まれています")

    for key in fm.legacy_keys:
        warnings.append(f"レガシーキー「{key}」が含まれています")

    return ValidationResult(ok=len(errors) == 0, warnings=warnings, errors=errors)


def _section_present(body: str, sec: str) -> bool:
    """body に sec のセクション見出しが存在するか確認する。"""
    # 完全一致: ## セクション名
    if f"## {sec}" in body:
        return True
    # 番号付き: ## N. セクション名
    if re.search(rf"^## \d+\. {re.escape(sec)}", body, re.MULTILINE):
        return True
    return False


def validate_sections(
    body: str,
    fm: PersonaFM,
    required_sections: tuple[str, ...] | None = None,
) -> ValidationResult:
    """
    ペルソナ本文のセクション検証。

    Args:
        body: ペルソナ本文
        fm: パース済みフロントマター
        required_sections: None → REQUIRED_SECTIONS を使用（後方互換）
                           () → セクション検証をスキップ
                           非 None のタプル → そのタプルで検証

    Returns:
        ValidationResult (ok, warnings, errors)
    """
    warnings: list[str] = []
    errors: list[str] = []

    # セクション検証をスキップ
    if required_sections is not None and len(required_sections) == 0:
        return ValidationResult(ok=True, warnings=warnings, errors=errors)

    sections = REQUIRED_SECTIONS if required_sections is None else required_sections

    for sec in sections:
        if not _section_present(body, sec):
            warnings.append(f"必須セクション「{sec}」が見つかりません")

    return ValidationResult(ok=len(errors) == 0, warnings=warnings, errors=errors)
