"""Tests that VALID_ENGINES stays in sync with ghdag ENGINE_SPECS."""
import pytest
from ghdag.llm.spec import ENGINE_SPECS
from mltgnt.persona.schema import VALID_ENGINES


def _expected() -> frozenset:
    return frozenset(ENGINE_SPECS.keys()) - {"shell"}


def test_valid_engines_equals_engine_specs_minus_shell():
    assert _expected() == VALID_ENGINES


def test_codex_in_valid_engines():
    assert "codex" in VALID_ENGINES


def test_shell_not_in_valid_engines():
    assert "shell" not in VALID_ENGINES


def test_valid_engines_is_frozenset():
    assert isinstance(VALID_ENGINES, frozenset)
