"""Every ghdag name referenced from ``mltgnt.bridges`` must exist upstream.

Collected from the ``bridges/**`` AST:
(a) ``from ghdag.X import N`` / ``import ghdag.X``
(b) attribute access ``N.attr`` on a ghdag-imported name
(c) ``getattr(<ghdag name>, "attr", default)`` - allowed only when listed in
    ``OPTIONAL_UPSTREAM_ATTRS`` (a silent fallback would hide upstream removals)
Additionally every ``on_*`` / ``check_*`` method of ``MltgntHooks`` must exist on ``DagHooks``.
"""
from __future__ import annotations

import ast
import importlib
import importlib.metadata
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGES_ROOT = REPO_ROOT / "src" / "mltgnt" / "bridges"

OPTIONAL_UPSTREAM_ATTRS: frozenset[str] = frozenset()

_HOOK_PREFIXES = ("on_", "check_")


def _is_ghdag(module: str | None) -> bool:
    return module is not None and (module == "ghdag" or module.startswith("ghdag."))


def _resolve_from(module: str, name: str) -> object:
    """Resolve ``from module import name`` (attribute first, then submodule)."""
    mod = importlib.import_module(module)
    if hasattr(mod, name):
        return getattr(mod, name)
    return importlib.import_module(f"{module}.{name}")


def _resolve_origin(origin: tuple[str, str | None]) -> object:
    module, name = origin
    return importlib.import_module(module) if name is None else _resolve_from(module, name)


def _dag_hooks_methods() -> set[str]:
    from ghdag.dag.hooks import DagHooks

    return {name for name in dir(DagHooks) if not name.startswith("_") and callable(getattr(DagHooks, name))}


def find_missing_upstream_names(bridges_root: Path) -> list[str]:
    """Return ``"<rel>:<line>: <reference>"`` for upstream references that do not resolve."""
    violations: list[str] = []
    dag_hooks_methods: set[str] | None = None
    for path in sorted(bridges_root.rglob("*.py")):
        rel = path.relative_to(bridges_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        # (a) imports; remember which local names come from ghdag.
        origins: dict[str, tuple[str, str | None]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and _is_ghdag(node.module):
                assert node.module is not None
                for alias in node.names:
                    try:
                        _resolve_from(node.module, alias.name)
                    except (ImportError, AttributeError):
                        violations.append(f"{rel}:{node.lineno}: from {node.module} import {alias.name}")
                        continue
                    origins[alias.asname or alias.name] = (node.module, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not _is_ghdag(alias.name):
                        continue
                    try:
                        importlib.import_module(alias.name)
                    except ImportError:
                        violations.append(f"{rel}:{node.lineno}: import {alias.name}")
                        continue
                    if alias.asname:
                        origins[alias.asname] = (alias.name, None)
                    else:
                        origins[alias.name.split(".")[0]] = ("ghdag", None)

        # (b) attribute access and (c) getattr with a default.
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in origins
                and isinstance(node.ctx, ast.Load)
            ):
                obj = _resolve_origin(origins[node.value.id])
                if not hasattr(obj, node.attr):
                    violations.append(f"{rel}:{node.lineno}: {node.value.id}.{node.attr}")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in origins
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                attr = node.args[1].value
                ref = f"{rel}:{node.lineno}: getattr({node.args[0].id}, {attr!r})"
                if len(node.args) >= 3:
                    if attr not in OPTIONAL_UPSTREAM_ATTRS:
                        violations.append(ref)
                elif not hasattr(_resolve_origin(origins[node.args[0].id]), attr):
                    violations.append(ref)

        # DagHooks compatibility of MltgntHooks.
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ClassDef) and node.name == "MltgntHooks"):
                continue
            if dag_hooks_methods is None:
                dag_hooks_methods = _dag_hooks_methods()
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name.startswith(_HOOK_PREFIXES)
                    and item.name not in dag_hooks_methods
                ):
                    violations.append(f"{rel}:{item.lineno}: MltgntHooks.{item.name} not in DagHooks")
    return violations


def _pinned_ghdag_version() -> str | None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"ghdag\.git@v([0-9][^\"'\s]*)", text)
    return match.group(1) if match else None


def test_real_bridges_reference_existing_upstream_names() -> None:
    pinned = _pinned_ghdag_version()
    installed = importlib.metadata.version("ghdag")
    if pinned is not None and installed != pinned:
        pytest.skip(f"installed ghdag {installed} does not match pinned v{pinned}")
    assert find_missing_upstream_names(BRIDGES_ROOT) == []


def test_hook_method_missing_from_dag_hooks_is_reported(tmp_path: Path) -> None:
    (tmp_path / "hooks.py").write_text(
        "class MltgntHooks:\n"
        "    def on_task_start(self, uuid, task):\n"
        "        return None\n"
        "\n"
        "    def on_task_nonexistent(self, uuid):\n"
        "        return None\n",
        encoding="utf-8",
    )
    assert find_missing_upstream_names(tmp_path) == ["hooks.py:5: MltgntHooks.on_task_nonexistent not in DagHooks"]


def test_missing_import_is_reported(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("from ghdag.dag.hooks import NoSuchName\n", encoding="utf-8")
    assert find_missing_upstream_names(tmp_path) == ["bad.py:1: from ghdag.dag.hooks import NoSuchName"]


def test_missing_attribute_is_reported(tmp_path: Path) -> None:
    (tmp_path / "attr.py").write_text(
        "from ghdag.dag.hooks import DagHooks\n"
        "\n"
        "X = DagHooks.on_task_start\n"
        "Y = DagHooks.no_such_attr\n",
        encoding="utf-8",
    )
    assert find_missing_upstream_names(tmp_path) == ["attr.py:4: DagHooks.no_such_attr"]


def test_getattr_with_default_requires_allowlist(tmp_path: Path) -> None:
    (tmp_path / "opt.py").write_text(
        "import ghdag.dag.hooks as hooks\n"
        "\n"
        "X = getattr(hooks, 'DagHooks', None)\n",
        encoding="utf-8",
    )
    assert find_missing_upstream_names(tmp_path) == ["opt.py:3: getattr(hooks, 'DagHooks')"]
