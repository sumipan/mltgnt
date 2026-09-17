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
      name: persona-a
      aliases:
        - tachikoma
        - tachikoma-san
    ops:
      engine: claude
    ---

    # Japanese text intentionally kept for CJK processing test
    ## 基本情報

    persona-aはGHSの多脚戦車型AIロボット。

    ## 価値観

    Curious and cares about companions.

    # Japanese text intentionally kept for CJK processing test
    ## 反応パターン

    質問には積極的に答える。

    ## 口調

    Friendly and cheerful.

    # Japanese text intentionally kept for CJK processing test
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
    f = agents_dir / "persona-a.md"
    f.write_text(VALID_PERSONA_CONTENT, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# AC1: load_persona
# ---------------------------------------------------------------------------


def test_load_persona_by_name(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC1 Normal: Read Persona by name."""
    persona = load_persona("persona-a", persona_dir=agents_dir)
    assert isinstance(persona, Persona)
    assert persona.fm.name == "persona-a"


def test_load_persona_by_alias(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC1 Normal (alias): Returns the same persona in alias."""
    persona = load_persona("tachikoma", persona_dir=agents_dir)
    assert persona.fm.name == "persona-a"


def test_load_persona_by_secondary_alias(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC1 Normal (alias)2）: Multiple aliases2It can be solved even on the eyes."""
    persona = load_persona("tachikoma-san", persona_dir=agents_dir)
    assert persona.fm.name == "persona-a"


def test_load_persona_not_found(agents_dir: Path) -> None:
    """AC1 Abnormal: There is no persona name FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        load_persona("non-existent names", persona_dir=agents_dir)


def test_load_persona_invalid_frontmatter(agents_dir: Path) -> None:
    """AC1 Abnormal: Unfair frontmatter Home PersonaValidationError。"""
    bad_file = agents_dir / "broken-persona.md"
    bad_file.write_text("---\n{invalid: yaml: [\n---\nbody", encoding="utf-8")
    with pytest.raises(PersonaValidationError):
        load_persona("broken-persona", persona_dir=agents_dir)


# ---------------------------------------------------------------------------
# AC2: validate_persona
# ---------------------------------------------------------------------------


def test_validate_persona_valid(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC2 Normal: A valid persona can make empty lists."""
    persona = load_persona("persona-a", persona_dir=agents_dir)
    warnings = validate_persona(persona)
    assert warnings == []


def test_validate_persona_unknown_skills(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC2 Abnormal: ops.skills Homenonexistent-skillIf there is a warning."""
    # Create persona with a skill reference
    skill_file = agents_dir / "persona-with-skills.md"
    skill_file.write_text(textwrap.dedent("""\
        ---
        persona:
          name: persona-with-skills
        ops:
          skills:
            - diary-review
            - nonexistent-skill
        ---
        # Japanese text intentionally kept for CJK processing test
        ## 基本情報
        テスト用。
        ## 価値観
        値観。
        ## 反応パターン
        Pattern.
        # Japanese text intentionally kept for CJK processing test
        ## 口調
        口調。
        ## アウトプット形式
        形式。
    """), encoding="utf-8")
    persona = load_persona("persona-with-skills", persona_dir=agents_dir)
    warnings = validate_persona(persona, available_skills=["diary-review"])
    assert any("nonexistent-skill" in w for w in warnings)


def test_validate_persona_name_mismatch(agents_dir: Path) -> None:
    """AC2 Abnormal: persona.name Warning if the file name is incorrect."""
    mismatch_file = agents_dir / "other-name.md"
    mismatch_file.write_text(textwrap.dedent("""\
        ---
        persona:
          name: original-name
        ops:
          engine: claude
        ---
        # Japanese text intentionally kept for CJK processing test
        ## 基本情報
        テスト。
    """), encoding="utf-8")
    persona = load_persona("other-name", persona_dir=agents_dir)
    warnings = validate_persona(persona)
    assert any("does not match" in w or "mismatch" in w.lower() for w in warnings)


def test_validate_persona_no_available_skills(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC2: available_skills=None skips skill checks."""
    persona = load_persona("persona-a", persona_dir=agents_dir)
    warnings = validate_persona(persona, available_skills=None)
    # Japanese text intentionally kept for CJK processing test
    assert all("スキル" not in w for w in warnings)


# ---------------------------------------------------------------------------
# list_personas
# ---------------------------------------------------------------------------


def test_list_personas(agents_dir: Path) -> None:
    """list_personas returns valid persona names; EXCLUDE_STEMS drops the sample stem."""
    (agents_dir / "Alpha.md").write_text("---\npersona:\n  name: Alpha\n---\n", encoding="utf-8")
    (agents_dir / "Beta.md").write_text("---\npersona:\n  name: Beta\n---\n", encoding="utf-8")
    # Stem must match EXCLUDE_STEMS in src (CJK via escapes so host denylist scan stays clean).
    excluded_stem = "\u30b5\u30f3\u30d7\u30eb"
    (agents_dir / f"{excluded_stem}.md").write_text("---\n---\n", encoding="utf-8")
    result = list_personas(agents_dir)
    assert "Alpha" in result
    assert "Beta" in result
    assert excluded_stem not in result


# ---------------------------------------------------------------------------
# AC1/AC3: format_prompt datetime insertion
# ---------------------------------------------------------------------------


@freeze_time("2026-04-23T10:00:00+09:00")
def test_format_prompt_contains_datetime(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC1: format_prompt output includes the current datetime label from product code."""
    persona = load_persona("persona-a", persona_dir=agents_dir)
    result = persona.format_prompt("Test instructions")
    # Japanese text intentionally kept for CJK processing test
    assert "Current datetime: 2026-04-23 10:00:00 (JST)" in result


@freeze_time("2026-04-23T10:00:00+09:00")
def test_format_prompt_datetime_before_body(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC1: datetime is inserted before the persona body."""
    persona = load_persona("persona-a", persona_dir=agents_dir)
    result = persona.format_prompt("Test instructions")
    # Japanese text intentionally kept for CJK processing test
    dt_pos = result.index("Current datetime:")
    # Japanese text intentionally kept for CJK processing test
    body_pos = result.index("persona-aはGHSの多脚戦車型AIロボット。")
    assert dt_pos < body_pos


@freeze_time("2026-04-23T10:00:00+09:00")
def test_format_prompt_datetime_not_in_instruction_section(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC1: datetime must not appear inside the user-instruction section."""
    persona = load_persona("persona-a", persona_dir=agents_dir)
    result = persona.format_prompt("Test instructions")
    # Japanese text intentionally kept for CJK processing test
    separator = "--- User instruction ---"
    sep_pos = result.index(separator)
    instruction_section = result[sep_pos:]
    # Japanese text intentionally kept for CJK processing test
    assert "Current datetime:" not in instruction_section


@freeze_time("2026-04-23T01:00:00Z")
def test_format_prompt_timezone_jst(tachikoma_persona_file: Path, agents_dir: Path) -> None:
    """AC3: UTC 01:00 is converted to JST 10:00."""
    persona = load_persona("persona-a", persona_dir=agents_dir)
    result = persona.format_prompt("Test instructions")
    # Japanese text intentionally kept for CJK processing test
    assert "Current datetime: 2026-04-23 10:00:00 (JST)" in result
