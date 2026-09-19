"""Tests for PersonaProtocol (issue-908, issue-1106)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from mltgnt.interfaces.persona import PersonaProtocol
from mltgnt.interfaces.types import PersonaFMBase
from mltgnt.persona.loader import Persona
from mltgnt.persona.schema import PersonaFM


VALID_PERSONA_CONTENT = textwrap.dedent("""\
    ---
    persona:
      name: test-persona
    ops:
      engine: claude
      model: claude-3-5-sonnet-20241022
    ---

    ## Basic information
    Persona for tests.

    ## Values
    Test.

    ## Reaction patterns
    Pattern.

    ## Tone
    Tone.

    ## Output format
    Format.
""")


@pytest.fixture
def persona(tmp_path: Path) -> Persona:
    d = tmp_path / "agents"
    d.mkdir()
    f = d / "test-persona.md"
    f.write_text(VALID_PERSONA_CONTENT, encoding="utf-8")
    from mltgnt.persona import load_persona
    return load_persona("test-persona", persona_dir=d)


def test_persona_isinstance_protocol(persona: Persona) -> None:
    """Persona instances satisfy PersonaProtocol."""
    assert isinstance(persona, PersonaProtocol)


def test_persona_fm_is_persona_fm(persona: Persona) -> None:
    """Persona.fm is a PersonaFM."""
    assert isinstance(persona.fm, PersonaFM)


def test_protocol_fm_access(persona: Persona) -> None:
    """.fm.engine is accessible via PersonaProtocol."""
    p: PersonaProtocol = persona
    assert p.fm.engine == "claude"


def test_no_fm_fails_isinstance() -> None:
    """Objects without fm fail isinstance."""
    class NoFM:
        name: str = "dummy"

        def format_prompt(self, instruction: str) -> str:
            return instruction

    assert not isinstance(NoFM(), PersonaProtocol)


def test_name_only_fails_isinstance() -> None:
    """Objects with only name also fail isinstance."""
    class NameOnly:
        name: str = "only-name"

        def format_prompt(self, instruction: str) -> str:
            return instruction

    assert not isinstance(NameOnly(), PersonaProtocol)


def test_persona_fm_satisfies_persona_fm_base(persona: Persona) -> None:
    """PersonaFM satisfies PersonaFMBase (structural subtyping)."""
    assert isinstance(persona.fm, PersonaFMBase)


def test_persona_protocol_no_domain_import() -> None:
    """interfaces/persona.py must not import mltgnt.persona.schema."""
    import importlib
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mltgnt.interfaces.persona_src",
        Path(__file__).parents[2] / "src" / "mltgnt" / "interfaces" / "persona.py",
    )
    assert spec is not None
    source_file = spec.origin
    assert source_file is not None
    content = Path(source_file).read_text()
    assert "mltgnt.persona" not in content, (
        "interfaces/persona.py must not import from mltgnt.persona (layer violation)"
    )
