"""Self check: tests under ``tests/conventions/`` must not use mocks.

Only LLM calls and clocks may be substituted, and those are passed as plain
arguments; all I/O goes to real files under ``tmp_path``.
"""
from __future__ import annotations

import ast
from pathlib import Path

CONVENTIONS_ROOT = Path(__file__).resolve().parent

_MOCK_MODULES = ("unittest.mock", "mock", "pytest_mock")


def _is_mock_module(name: str | None) -> bool:
    return name is not None and any(name == m or name.startswith(m + ".") for m in _MOCK_MODULES)


def find_mock_usage(conventions_root: Path) -> list[str]:
    """Return ``"<file>:<line>: <what>"`` for mock imports and ``mocker`` fixtures."""
    violations: list[str] = []
    for path in sorted(conventions_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_mock_module(alias.name):
                        violations.append(f"{path.name}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [alias.name for alias in node.names]
                if _is_mock_module(node.module) or (node.module == "unittest" and "mock" in names):
                    violations.append(f"{path.name}:{node.lineno}: from {node.module} import {', '.join(names)}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                all_args = [*args.posonlyargs, *args.args, *args.kwonlyargs]
                if any(arg.arg == "mocker" for arg in all_args):
                    violations.append(f"{path.name}:{node.lineno}: mocker fixture in {node.name}")
    return violations


def test_conventions_do_not_use_mocks() -> None:
    assert find_mock_usage(CONVENTIONS_ROOT) == []


def test_mock_import_is_reported(tmp_path: Path) -> None:
    (tmp_path / "test_bad.py").write_text(
        "from unittest.mock import MagicMock\n"
        "from unittest import mock\n"
        "import pytest_mock\n"
        "\n"
        "\n"
        "def test_x(mocker):\n"
        "    pass\n",
        encoding="utf-8",
    )
    assert find_mock_usage(tmp_path) == [
        "test_bad.py:1: from unittest.mock import MagicMock",
        "test_bad.py:2: from unittest import mock",
        "test_bad.py:3: import pytest_mock",
        "test_bad.py:6: mocker fixture in test_x",
    ]
