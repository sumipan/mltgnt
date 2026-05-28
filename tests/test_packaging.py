"""PEP 561 packaging: py.typed marker is included in the distribution."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import mltgnt


def test_py_typed_via_importlib_resources() -> None:
    marker = importlib.resources.files("mltgnt") / "py.typed"
    assert marker.is_file(), "py.typed missing in mltgnt package (importlib.resources)"


def test_py_typed_via_package_path() -> None:
    marker = Path(mltgnt.__file__).parent / "py.typed"
    assert marker.is_file(), "py.typed missing next to mltgnt package (pathlib)"
