"""Only ``mltgnt.bridges`` may import ghdag; no host-specific strings in ``src/``.

``secretary`` style words are part of the configuration API (e.g. the
``notify`` values in ``scheduler.models``) and are intentionally not checked.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mltgnt"

HOST_TOKENS = ("/Users/", "Github/nexus", '"jobs/', "ghdag_watcher")

KNOWN_HOST_LEAK_FILES = ["bridges/ghdag_bridge.py"]


def _is_ghdag(module: str | None) -> bool:
    return module is not None and (module == "ghdag" or module.startswith("ghdag."))


def find_ghdag_imports_outside_bridges(src_root: Path) -> list[str]:
    """Return ``"<rel>:<line>: <source>"`` for ghdag imports outside ``bridges/`` (lazy imports included)."""
    bridges = src_root / "bridges"
    violations: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        if bridges in path.parents:
            continue
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if isinstance(node, ast.Import):
                hit = any(_is_ghdag(alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                hit = node.level == 0 and _is_ghdag(node.module)
            else:
                continue
            if hit:
                rel = path.relative_to(src_root).as_posix()
                violations.append(f"{rel}:{node.lineno}: {lines[node.lineno - 1].strip()}")
    return sorted(violations)


def find_host_leaks(src_root: Path, tokens: tuple[str, ...] | list[str]) -> list[str]:
    """Return ``"<rel>:<line>: <line text>"`` for lines containing any host-specific token."""
    violations: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        rel = path.relative_to(src_root).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(token in line for token in tokens):
                violations.append(f"{rel}:{lineno}: {line.strip()}")
    return violations


def _file_of(violation: str) -> str:
    return violation.split(":", 1)[0]


def test_no_ghdag_imports_outside_bridges() -> None:
    assert find_ghdag_imports_outside_bridges(SRC_ROOT) == []


def test_no_unexpected_host_leaks() -> None:
    leaks = find_host_leaks(SRC_ROOT, HOST_TOKENS)
    assert [v for v in leaks if _file_of(v) not in KNOWN_HOST_LEAK_FILES] == []


@pytest.mark.parametrize("rel_file", KNOWN_HOST_LEAK_FILES)
@pytest.mark.xfail(strict=True, reason="#3573 known violation: host-specific 'jobs/' in ghdag_bridge docstring")
def test_known_host_leak_removed(rel_file: str) -> None:
    assert [v for v in find_host_leaks(SRC_ROOT, HOST_TOKENS) if _file_of(v) == rel_file] == []


def test_ghdag_import_outside_bridges_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "src" / "mltgnt"
    (root / "persona").mkdir(parents=True)
    (root / "bridges").mkdir()
    (root / "persona" / "x.py").write_text("from ghdag.files import md_read\n", encoding="utf-8")
    (root / "persona" / "lazy.py").write_text("def f():\n    import ghdag.llm\n", encoding="utf-8")
    (root / "bridges" / "ok.py").write_text("from ghdag.files import md_read\n", encoding="utf-8")
    violations = find_ghdag_imports_outside_bridges(root)
    assert "persona/x.py:1: from ghdag.files import md_read" in violations
    assert "persona/lazy.py:2: import ghdag.llm" in violations
    assert len(violations) == 2


def test_host_leak_is_reported(tmp_path: Path) -> None:
    (tmp_path / "leak.py").write_text("PATH = '/Users/someone/Github/nexus'\n", encoding="utf-8")
    assert find_host_leaks(tmp_path, HOST_TOKENS) == ["leak.py:1: PATH = '/Users/someone/Github/nexus'"]
