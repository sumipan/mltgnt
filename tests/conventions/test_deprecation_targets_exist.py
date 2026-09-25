"""Deprecation warnings must point at a replacement that actually exists.

For every ``warnings.warn(..., DeprecationWarning | FutureWarning)`` the
``use <name>`` target in the message must be resolvable via ``getattr`` on the
same module (or, failing that, its package).
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from mltgnt.routing.channel_router import find_observers, resolve_responding_persona

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mltgnt"

_DEPRECATION_CATEGORIES = {"DeprecationWarning", "FutureWarning"}
_USE_RE = re.compile(r"\buse\s+`*([A-Za-z_][A-Za-z0-9_.]*)")


def _is_warn_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == "warn" and isinstance(func.value, ast.Name) and func.value.id == "warnings"
    return isinstance(func, ast.Name) and func.id == "warn"


def _category_name(node: ast.Call) -> str | None:
    category: ast.expr | None = node.args[1] if len(node.args) > 1 else None
    for kw in node.keywords:
        if kw.arg == "category":
            category = kw.value
    if isinstance(category, ast.Name):
        return category.id
    if isinstance(category, ast.Attribute):
        return category.attr
    return None


def _message_text(node: ast.Call) -> str:
    message: ast.expr | None = node.args[0] if node.args else None
    for kw in node.keywords:
        if kw.arg == "message":
            message = kw.value
    if isinstance(message, ast.Constant) and isinstance(message.value, str):
        return message.value
    if isinstance(message, ast.JoinedStr):
        return "".join(
            part.value for part in message.values if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    return ""


def _load_module(path: Path, src_root: Path) -> ModuleType:
    """Import ``path``; prefer the regular import system, fall back to loading by file."""
    rel = path.relative_to(src_root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    dotted = ".".join(parts)
    try:
        module = importlib.import_module(dotted)
        if module.__file__ and Path(module.__file__).resolve() == path.resolve():
            return module
    except ImportError:
        pass
    name = "_conventions_loaded_" + re.sub(r"\W", "_", str(path.resolve()))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _resolves(target: str, modules: list[ModuleType]) -> bool:
    for module in modules:
        obj: object = module
        try:
            for attr in target.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return True
    return False


def find_missing_replacements(src_root: Path) -> list[str]:
    """Return ``"<rel>:<line>: use <name>"`` for replacements that cannot be resolved."""
    violations: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_warn_call(node):
                continue
            if _category_name(node) not in _DEPRECATION_CATEGORIES:
                continue
            for target in _USE_RE.findall(_message_text(node)):
                target = target.rstrip(".")
                modules = [_load_module(path, src_root)]
                package_init = path.parent / "__init__.py"
                if package_init.exists() and package_init != path:
                    modules.append(_load_module(package_init, src_root))
                if not _resolves(target, modules):
                    rel = path.relative_to(src_root).as_posix()
                    violations.append(f"{rel}:{node.lineno}: use {target}")
    return violations


def test_real_replacements_exist() -> None:
    assert find_missing_replacements(SRC_ROOT) == []


def test_resolve_responding_persona_warns() -> None:
    with pytest.warns(DeprecationWarning, match="use resolve_persona"):
        resolve_responding_persona("C1", "hello", None, {}, {})


def test_find_observers_warns() -> None:
    with pytest.warns(DeprecationWarning, match="use find_observers_in_space"):
        assert find_observers("C1", None, {}) == []


def test_missing_replacement_is_reported(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "mltgnt" / "conv_fixture_deprecation"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "old.py").write_text(
        "import warnings\n"
        "\n"
        "\n"
        "def old_fn():\n"
        "    warnings.warn('old_fn is deprecated; use missing_fn', DeprecationWarning, stacklevel=2)\n"
        "\n"
        "\n"
        "def other_fn():\n"
        "    warnings.warn('other_fn is deprecated; use old_fn', FutureWarning, stacklevel=2)\n",
        encoding="utf-8",
    )
    violations = find_missing_replacements(tmp_path / "src" / "mltgnt")
    assert violations == ["conv_fixture_deprecation/old.py:5: use missing_fn"]
