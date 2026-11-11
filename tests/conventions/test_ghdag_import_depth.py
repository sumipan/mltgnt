"""Deep ghdag import paths and shim modules must not appear in src/ or tests/."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_SHIM_MODULES = frozenset({"ghdag.exceptions", "ghdag.github_client"})


def _is_deep_or_shim_module(module: str) -> bool:
    if module in _SHIM_MODULES:
        return True
    parts = module.split(".")
    if not parts or parts[0] != "ghdag":
        return False
    return len(parts) >= 3


def _module_from_dotted_target(target: str) -> str:
    parts = target.split(".")
    if len(parts) <= 1:
        return target
    return ".".join(parts[:-1])


def _importlib_module_literal(call: ast.Call) -> str | None:
    if not call.args:
        return None
    func = call.func
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "import_module"
        and isinstance(func.value, ast.Name)
        and func.value.id == "importlib"
    ):
        return _string_literal(call.args[0])
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "find_spec"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "util"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "importlib"
    ):
        return _string_literal(call.args[0])
    return None


def _string_literal(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_is_patch(func: ast.expr) -> bool:
    return (isinstance(func, ast.Name) and func.id == "patch") or (
        isinstance(func, ast.Attribute) and func.attr == "patch"
    )


def _check_string_module_ref(
    violations: list[str],
    *,
    rel: str,
    lineno: int,
    value: str,
) -> None:
    module = _module_from_dotted_target(value) if "." in value else value
    if _is_deep_or_shim_module(module):
        violations.append(f"{rel}:{lineno}: {value}")


def _visit_tree(tree: ast.AST, rel: str, violations: list[str]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (alias.name == "ghdag" or alias.name.startswith("ghdag.")) and _is_deep_or_shim_module(
                    alias.name
                ):
                    violations.append(f"{rel}:{node.lineno}: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if (node.module == "ghdag" or node.module.startswith("ghdag.")) and _is_deep_or_shim_module(
                node.module
            ):
                violations.append(f"{rel}:{node.lineno}: {node.module}")
        elif isinstance(node, ast.Call):
            if _call_is_patch(node.func) and node.args:
                literal = _string_literal(node.args[0])
                if literal is not None:
                    _check_string_module_ref(
                        violations, rel=rel, lineno=node.lineno, value=literal
                    )
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "setattr" and node.args:
                literal = _string_literal(node.args[0])
                if literal is not None:
                    _check_string_module_ref(
                        violations, rel=rel, lineno=node.lineno, value=literal
                    )
            importlib_literal = _importlib_module_literal(node)
            if importlib_literal is not None and _is_deep_or_shim_module(importlib_literal):
                violations.append(f"{rel}:{node.lineno}: {importlib_literal}")


def find_deep_ghdag_references(roots: list[Path], base: Path) -> list[str]:
    """Return violation strings for deep ghdag imports and disallowed string targets."""
    violations: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(base).as_posix()
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            _visit_tree(tree, rel, violations)
    return sorted(violations)


def test_src_and_tests_have_no_deep_ghdag_references() -> None:
    roots = [REPO_ROOT / "src", REPO_ROOT / "tests"]
    assert find_deep_ghdag_references(roots, REPO_ROOT) == []


def test_detects_deep_import(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("from ghdag.pipeline.state import PipelineState\n", encoding="utf-8")
    assert find_deep_ghdag_references([tmp_path], tmp_path) == [
        "sample.py:1: ghdag.pipeline.state"
    ]


def test_detects_shim_import(tmp_path: Path) -> None:
    sample = tmp_path / "shim.py"
    sample.write_text(
        "from ghdag.exceptions import GitHubApiError\n"
        "import ghdag.github_client\n",
        encoding="utf-8",
    )
    found = find_deep_ghdag_references([tmp_path], tmp_path)
    assert found == [
        "shim.py:1: ghdag.exceptions",
        "shim.py:2: ghdag.github_client",
    ]


def test_detects_deep_string_target_and_allows_one_level(tmp_path: Path) -> None:
    sample = tmp_path / "patch_targets.py"
    sample.write_text(
        'patch("ghdag.pipeline.state.PipelineState")\n'
        'patch("ghdag.pipeline.wait_for_result")\n'
        "from ghdag.io import done\n"
        "from ghdag.dag import DagHooks\n",
        encoding="utf-8",
    )
    assert find_deep_ghdag_references([tmp_path], tmp_path) == [
        "patch_targets.py:1: ghdag.pipeline.state.PipelineState"
    ]
