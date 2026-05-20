"""Tests for PersonaProtocol (issue-908)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from mltgnt.interfaces.persona import PersonaProtocol
from mltgnt.persona.loader import Persona
from mltgnt.persona.schema import PersonaFM


VALID_PERSONA_CONTENT = textwrap.dedent("""\
    ---
    persona:
      name: テストペルソナ
    ops:
      engine: claude
      model: claude-3-5-sonnet-20241022
    ---

    ## 基本情報
    テスト用ペルソナ。

    ## 価値観
    テスト。

    ## 反応パターン
    パターン。

    ## 口調
    口調。

    ## アウトプット形式
    形式。
""")


@pytest.fixture
def persona(tmp_path: Path) -> Persona:
    d = tmp_path / "agents"
    d.mkdir()
    f = d / "テストペルソナ.md"
    f.write_text(VALID_PERSONA_CONTENT, encoding="utf-8")
    from mltgnt.persona import load_persona
    return load_persona("テストペルソナ", persona_dir=d)


def test_persona_isinstance_protocol(persona: Persona) -> None:
    """Persona インスタンスは PersonaProtocol を満たす。"""
    assert isinstance(persona, PersonaProtocol)


def test_persona_fm_is_persona_fm(persona: Persona) -> None:
    """Persona.fm が PersonaFM 型である。"""
    assert isinstance(persona.fm, PersonaFM)


def test_protocol_fm_access(persona: Persona) -> None:
    """PersonaProtocol 型経由で .fm.engine にアクセスできる。"""
    p: PersonaProtocol = persona
    assert p.fm.engine == "claude"


def test_no_fm_fails_isinstance() -> None:
    """fm プロパティを持たないオブジェクトは isinstance が False。"""
    class NoFM:
        name: str = "dummy"

        def format_prompt(self, instruction: str) -> str:
            return instruction

    assert not isinstance(NoFM(), PersonaProtocol)


def test_name_only_fails_isinstance() -> None:
    """name のみ持ち fm を持たないオブジェクトも isinstance が False。"""
    class NameOnly:
        name: str = "only-name"

        def format_prompt(self, instruction: str) -> str:
            return instruction

    assert not isinstance(NameOnly(), PersonaProtocol)
