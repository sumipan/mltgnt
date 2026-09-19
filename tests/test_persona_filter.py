"""Tests for Persona.register_prompt_filter API."""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
from freezegun import freeze_time

from mltgnt.persona import load_persona
from mltgnt.persona.loader import Persona, PromptFilter
from mltgnt.interfaces.persona import PersonaProtocol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_PERSONA_CONTENT = textwrap.dedent("""\
    ---
    persona:
      name: persona-a
      aliases:
        - tachikoma
    ops:
      engine: claude
    ---

    ## Basic information

    persona-a is a multi-legged tank-type AI robot from GHS.

    ## Values

    Curious and values companions.

    ## Reaction patterns

    Answers questions eagerly.

    ## Tone

    Friendly and cheerful.
""")


@pytest.fixture
def agents_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agents"
    d.mkdir()
    return d


@pytest.fixture
def tachikoma_persona(agents_dir: Path) -> Persona:
    (agents_dir / "persona-a.md").write_text(VALID_PERSONA_CONTENT, encoding="utf-8")
    return load_persona("persona-a", persona_dir=agents_dir)


# ---------------------------------------------------------------------------
# AC: register_prompt_filter → reflected in format_prompt
# ---------------------------------------------------------------------------


def test_register_custom_filter_output_appears(tachikoma_persona: Persona) -> None:
    """Custom filter return value appears in format_prompt output."""
    def custom_fn(accumulated: str, ctx: dict[str, Any]) -> str:
        return accumulated + "CUSTOM_PREFIX_LINE\n\n"

    tachikoma_persona.register_prompt_filter("custom", custom_fn)
    result = tachikoma_persona.format_prompt("test")
    assert "CUSTOM_PREFIX_LINE" in result


def test_replace_datetime_filter(tachikoma_persona: Persona) -> None:
    """Replacing the datetime filter removes the old default 'Current datetime:' line."""
    def new_fn(accumulated: str, ctx: dict[str, Any]) -> str:
        return accumulated + "REPLACED_DATETIME\n\n"

    tachikoma_persona.register_prompt_filter("datetime", new_fn)
    result = tachikoma_persona.format_prompt("test")
    assert "REPLACED_DATETIME" in result
    assert "Current datetime:" not in result


@freeze_time("2026-04-23T10:00:00+09:00")
def test_default_datetime_filter_backward_compat(tachikoma_persona: Persona) -> None:
    """Without register_prompt_filter, the existing 'Current datetime:' line is still present."""
    result = tachikoma_persona.format_prompt("test")
    assert "Current datetime: 2026-04-23 10:00:00 (JST)" in result


def test_multiple_filters_ordered_accumulation(tachikoma_persona: Persona) -> None:
    """Multiple filters accumulate prefixes in registration order."""
    calls: list[str] = []

    def filter_a(accumulated: str, ctx: dict[str, Any]) -> str:
        calls.append("a")
        return accumulated + "AAA\n\n"

    def filter_b(accumulated: str, ctx: dict[str, Any]) -> str:
        calls.append("b")
        assert "AAA" in accumulated  # a's output is passed to b
        return accumulated + "BBB\n\n"

    # Replace datetime to keep the chain simple
    tachikoma_persona.register_prompt_filter("datetime", filter_a)
    tachikoma_persona.register_prompt_filter("extra", filter_b)
    result = tachikoma_persona.format_prompt("test")

    assert calls == ["a", "b"]
    assert "AAA" in result
    assert "BBB" in result
    aaa_pos = result.index("AAA")
    bbb_pos = result.index("BBB")
    assert aaa_pos < bbb_pos


def test_protocol_has_register_prompt_filter() -> None:
    """PersonaProtocol defines register_prompt_filter."""
    assert hasattr(PersonaProtocol, "register_prompt_filter")


def test_persona_satisfies_protocol(tachikoma_persona: Persona) -> None:
    """Persona satisfies PersonaProtocol."""
    assert isinstance(tachikoma_persona, PersonaProtocol)
