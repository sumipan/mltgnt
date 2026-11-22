"""pyproject.toml ghdag dependency shape (Issue #7795)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def test_pyproject_ghdag_is_single_lower_bound() -> None:
    """The ghdag dependency is exactly one lower-bound specifier without a URL."""
    specs = [
        dep
        for dep in _pyproject()["project"]["dependencies"]
        if dep.startswith("ghdag")
    ]
    assert len(specs) == 1, f"expected exactly one ghdag dependency: {specs}"
    assert re.fullmatch(r"ghdag>=\d+\.\d+\.\d+", specs[0]), (
        f"ghdag dependency is not a single lower bound: {specs[0]}"
    )
