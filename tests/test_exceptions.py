"""tests/test_exceptions.py — 例外型階層のユニットテスト（Issue #1252 AC-2）。"""
from __future__ import annotations

from mltgnt.exceptions import ConfigError, DependencyError, MltgntError


def test_exception_hierarchy() -> None:
    assert issubclass(ConfigError, MltgntError)
    assert issubclass(DependencyError, MltgntError)
    assert issubclass(MltgntError, Exception)
