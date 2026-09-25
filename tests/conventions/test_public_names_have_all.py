"""``__all__`` must be importable and public; private names must not cross packages.

The ``__all__`` snapshot itself lives in ``tests/test_all_snapshot.py``.
"""
from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

import mltgnt

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mltgnt"

_MEMORY_REASON = "#3573 known violation: private names in mltgnt.memory.__all__"
_SCHEDULER_REASON = "#3573 known violation: private names in mltgnt.scheduler.__all__"

KNOWN_PRIVATE_IN_ALL = {
    "mltgnt.memory: _ensure_jsonl": _MEMORY_REASON,
    "mltgnt.memory: _resolve_memory_dir": _MEMORY_REASON,
    "mltgnt.memory: _scan_tail_for_dedupe_key": _MEMORY_REASON,
    "mltgnt.memory: _search_and_score": _MEMORY_REASON,
    "mltgnt.scheduler: _hash_offset": _SCHEDULER_REASON,
}


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _all_names(pkg_name: str) -> list[str]:
    module = importlib.import_module(pkg_name)
    return list(getattr(module, "__all__", []))


def mltgnt_packages() -> list[str]:
    """``mltgnt`` and each package directly below it."""
    names = ["mltgnt"]
    names += sorted(f"mltgnt.{info.name}" for info in pkgutil.iter_modules(mltgnt.__path__) if info.ispkg)
    return names


def find_unimportable_all(pkg_names: list[str]) -> list[str]:
    """Return ``"<pkg>: <name>"`` for ``__all__`` entries that ``getattr`` cannot resolve."""
    violations: list[str] = []
    for pkg_name in pkg_names:
        module = importlib.import_module(pkg_name)
        for name in _all_names(pkg_name):
            if not hasattr(module, name):
                violations.append(f"{pkg_name}: {name}")
    return violations


def find_private_in_all(pkg_names: list[str]) -> list[str]:
    """Return ``"<pkg>: <name>"`` for ``_``-prefixed (non-dunder) names in ``__all__``."""
    return [f"{pkg_name}: {name}" for pkg_name in pkg_names for name in _all_names(pkg_name) if _is_private(name)]


def _top_package(parts: tuple[str, ...]) -> str:
    """Top-level package of a module path relative to the root (``""`` for root modules)."""
    return parts[0] if len(parts) > 1 else ""


def _absolute_module(node: ast.ImportFrom, rel_parts: tuple[str, ...], root_name: str) -> str:
    if node.level == 0:
        return node.module or ""
    package = [root_name, *rel_parts[:-1]]
    if node.level > 1:
        package = package[: len(package) - (node.level - 1)]
    return ".".join([*package, node.module] if node.module else package)


def find_cross_package_private_imports(src_root: Path) -> list[str]:
    """Return imports of ``_``-prefixed names from a different top-level package."""
    root_name = src_root.name
    violations: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        rel_parts = path.relative_to(src_root).parts
        own_top = _top_package(rel_parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = _absolute_module(node, rel_parts, root_name)
            mod_parts = module.split(".")
            if mod_parts[0] != root_name:
                continue
            target_top = mod_parts[1] if len(mod_parts) > 1 else ""
            if target_top == own_top:
                continue
            for alias in node.names:
                if _is_private(alias.name):
                    rel = path.relative_to(src_root).as_posix()
                    violations.append(f"{rel}:{node.lineno}: from {module} import {alias.name}")
    return violations


def test_all_names_are_importable() -> None:
    assert find_unimportable_all(mltgnt_packages()) == []


def test_no_unexpected_private_names_in_all() -> None:
    unexpected = [v for v in find_private_in_all(mltgnt_packages()) if v not in KNOWN_PRIVATE_IN_ALL]
    assert unexpected == []


@pytest.mark.parametrize(
    "violation",
    [
        pytest.param(violation, marks=pytest.mark.xfail(strict=True, reason=reason), id=violation)
        for violation, reason in KNOWN_PRIVATE_IN_ALL.items()
    ],
)
def test_known_private_name_removed_from_all(violation: str) -> None:
    assert violation not in find_private_in_all(mltgnt_packages())


def test_no_cross_package_private_imports() -> None:
    assert find_cross_package_private_imports(SRC_ROOT) == []


def test_private_name_in_all_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "conv_fixture_private_all"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("_x = 1\n__all__ = ['_x']\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert find_private_in_all(["conv_fixture_private_all"]) == ["conv_fixture_private_all: _x"]


def test_unimportable_name_in_all_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "conv_fixture_unimportable_all"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("__all__ = ['missing']\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert find_unimportable_all(["conv_fixture_unimportable_all"]) == ["conv_fixture_unimportable_all: missing"]


def test_cross_package_private_import_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "mltgnt"
    (root / "memory").mkdir(parents=True)
    (root / "persona").mkdir()
    (root / "memory" / "__init__.py").write_text("_hidden = 1\n", encoding="utf-8")
    (root / "memory" / "inner.py").write_text("from mltgnt.memory import _hidden\n", encoding="utf-8")
    (root / "persona" / "x.py").write_text("from mltgnt.memory import _hidden\n", encoding="utf-8")
    assert find_cross_package_private_imports(root) == ["persona/x.py:1: from mltgnt.memory import _hidden"]
