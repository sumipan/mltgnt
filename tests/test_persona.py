"""Tests for mltgnt.persona module (AC1, AC2)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from freezegun import freeze_time

from mltgnt.persona import (
    PersonaValidationError,
    list_personas,
    load_persona,
    validate_persona,
)
from mltgnt.persona.loader import Persona


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_PERSONA_CONTENT = textwrap.dedent("""\
    ---
    persona:
      name: タチコマ
      aliases:
        - tachikoma
        - tachikoma-san
    ops:
      engine: claude
    ---

    ## 基本情報

    タチコマはGHSの多脚戦車型AIロボット。

    ## 価値観

    好奇心旺盛で仲間を大切にする。

    ## 反応パターン

    質問には積極的に答える。

    ## 口調

    フレンドリーで明るい。

    ## アウトプット形式

    箇条書きを好む。
""")


@pytest.fixture
def agents_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agents"
    d.mkdir()
    return d


@pytest.fixture
def tachikoma_persona_file(agents_dir: Path) -> Path:
    f = agents_dir / "タチコマ.md"
    f.write_text(VALID_PERSONA_CONTENT, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# AC1: load_persona
# ---------------------------------------------------------------------------


def test_load_persona_by_name(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC1 正常系: 名前でペルソナを読み込む。"""
    persona = load_persona("タチコマ", persona_dir=agents_dir)
    assert isinstance(persona, Persona)
    assert persona.fm.name == "タチコマ"


def test_load_persona_by_alias(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC1 正常系（エイリアス）: エイリアスで同一ペルソナを返す。"""
    persona = load_persona("tachikoma", persona_dir=agents_dir)
    assert persona.fm.name == "タチコマ"


def test_load_persona_by_secondary_alias(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC1 正常系（エイリアス2）: 複数エイリアスの2番目でも解決できる。"""
    persona = load_persona("tachikoma-san", persona_dir=agents_dir)
    assert persona.fm.name == "タチコマ"


def test_load_persona_not_found(agents_dir: Path) -> None:
    """AC1 異常系: 存在しないペルソナ名は FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        load_persona("存在しない名前", persona_dir=agents_dir)


def test_load_persona_invalid_frontmatter(agents_dir: Path) -> None:
    """AC1 異常系: 不正な frontmatter は PersonaValidationError。"""
    bad_file = agents_dir / "壊れたペルソナ.md"
    bad_file.write_text("---\n{invalid: yaml: [\n---\n本文", encoding="utf-8")
    with pytest.raises(PersonaValidationError):
        load_persona("壊れたペルソナ", persona_dir=agents_dir)


# ---------------------------------------------------------------------------
# AC2: validate_persona
# ---------------------------------------------------------------------------


def test_validate_persona_valid(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC2 正常系: 有効なペルソナは空リストを返す。"""
    persona = load_persona("タチコマ", persona_dir=agents_dir)
    warnings = validate_persona(persona)
    assert warnings == []


def test_validate_persona_unknown_skills(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC2 異常系: ops.skills に存在しないスキルがあれば警告。"""
    # Create persona with a skill reference
    skill_file = agents_dir / "スキルあり.md"
    skill_file.write_text(textwrap.dedent("""\
        ---
        persona:
          name: スキルあり
        ops:
          skills:
            - diary-review
            - 存在しないスキル
        ---
        ## 基本情報
        テスト用。
        ## 価値観
        値観。
        ## 反応パターン
        パターン。
        ## 口調
        口調。
        ## アウトプット形式
        形式。
    """), encoding="utf-8")
    persona = load_persona("スキルあり", persona_dir=agents_dir)
    warnings = validate_persona(persona, available_skills=["diary-review"])
    assert any("存在しないスキル" in w for w in warnings)


def test_validate_persona_name_mismatch(agents_dir: Path) -> None:
    """AC2 異常系: persona.name とファイル名が不一致なら警告。"""
    mismatch_file = agents_dir / "別の名前.md"
    mismatch_file.write_text(textwrap.dedent("""\
        ---
        persona:
          name: 元の名前
        ops:
          engine: claude
        ---
        ## 基本情報
        テスト。
    """), encoding="utf-8")
    persona = load_persona("別の名前", persona_dir=agents_dir)
    warnings = validate_persona(persona)
    assert any("不一致" in w or "mismatch" in w.lower() for w in warnings)


def test_validate_persona_no_available_skills(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC2: available_skills=None のとき skill チェックをスキップ。"""
    persona = load_persona("タチコマ", persona_dir=agents_dir)
    warnings = validate_persona(persona, available_skills=None)
    # skill チェックをスキップするため、skills 関連の警告は含まれない
    assert all("スキル" not in w for w in warnings)


# ---------------------------------------------------------------------------
# list_personas
# ---------------------------------------------------------------------------


def test_list_personas(agents_dir: Path) -> None:
    """list_personas は有効なペルソナ名のリストを返す。"""
    (agents_dir / "Alpha.md").write_text("---\npersona:\n  name: Alpha\n---\n", encoding="utf-8")
    (agents_dir / "Beta.md").write_text("---\npersona:\n  name: Beta\n---\n", encoding="utf-8")
    (agents_dir / "サンプル.md").write_text("---\n---\n", encoding="utf-8")  # excluded
    result = list_personas(agents_dir)
    assert "Alpha" in result
    assert "Beta" in result
    assert "サンプル" not in result


# ---------------------------------------------------------------------------
# AC1/AC3: format_prompt datetime insertion
# ---------------------------------------------------------------------------


@freeze_time("2026-04-23T10:00:00+09:00")
def test_format_prompt_contains_datetime(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC1 正常系: format_prompt の出力に現在日時が含まれる。"""
    persona = load_persona("タチコマ", persona_dir=agents_dir)
    result = persona.format_prompt("テスト指示")
    assert "現在日時: 2026-04-23 10:00:00 (JST)" in result


@freeze_time("2026-04-23T10:00:00+09:00")
def test_format_prompt_datetime_before_body(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC1 正常系: 日時はペルソナ本文（body）の前に挿入される。"""
    persona = load_persona("タチコマ", persona_dir=agents_dir)
    result = persona.format_prompt("テスト指示")
    dt_pos = result.index("現在日時:")
    body_pos = result.index(persona.body)
    assert dt_pos < body_pos


@freeze_time("2026-04-23T10:00:00+09:00")
def test_format_prompt_datetime_not_in_instruction_section(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC1 正常系: --- ユーザーからの指示 --- セクションに日時が混入しない。"""
    persona = load_persona("タチコマ", persona_dir=agents_dir)
    result = persona.format_prompt("テスト指示")
    separator = "--- ユーザーからの指示 ---"
    sep_pos = result.index(separator)
    instruction_section = result[sep_pos:]
    assert "現在日時:" not in instruction_section


@freeze_time("2026-04-23T01:00:00Z")
def test_format_prompt_timezone_jst(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC3 正常系: UTC 01:00 → JST 10:00 に変換される。"""
    persona = load_persona("タチコマ", persona_dir=agents_dir)
    result = persona.format_prompt("テスト指示")
    assert "現在日時: 2026-04-23 10:00:00 (JST)" in result
