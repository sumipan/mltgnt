"""tests/test_exceptions.py — exception hierarchy unit tests (Issue #1252 AC-2)."""
from __future__ import annotations

from mltgnt.exceptions import ConfigError, DependencyError, MltgntError


def test_exception_hierarchy() -> None:
    assert issubclass(ConfigError, MltgntError)
    assert issubclass(DependencyError, MltgntError)
    assert issubclass(MltgntError, Exception)
