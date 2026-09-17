"""Acceptance tests for #3321 / #3301 sub6 — unused module removal."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src" / "mltgnt"

_REMOVED_PACKAGES = (
    "mltgnt.loops",
    "mltgnt.ooda",
    "mltgnt.improvement",
    "mltgnt.kpi",
    "mltgnt.chat",
    "mltgnt.execution",
)

_REMOVED_INTERFACE_MODULES = (
    "mltgnt.interfaces.loops",
    "mltgnt.interfaces.ooda",
    "mltgnt.interfaces.chat",
)

_REMOVED_PATHS = (
    "loops",
    "ooda",
    "improvement",
    "kpi",
    "chat",
    "execution",
)

_REMOVED_INTERFACE_FILES = (
    "interfaces/loops.py",
    "interfaces/ooda.py",
    "interfaces/chat.py",
)


@pytest.mark.parametrize("module_path", _REMOVED_PACKAGES)
def test_removed_packages_raise_import_error(module_path: str) -> None:
    with pytest.raises(ImportError):
        importlib.import_module(module_path)


@pytest.mark.parametrize("module_path", _REMOVED_INTERFACE_MODULES)
def test_removed_interface_modules_raise_import_error(module_path: str) -> None:
    with pytest.raises(ImportError):
        importlib.import_module(module_path)


@pytest.mark.parametrize("rel", _REMOVED_PATHS)
def test_removed_package_directories_absent(rel: str) -> None:
    assert not (_SRC / rel).exists()


@pytest.mark.parametrize("rel", _REMOVED_INTERFACE_FILES)
def test_removed_interface_files_absent(rel: str) -> None:
    assert not (_SRC / rel).exists()


def test_top_level_no_longer_exports_run_pipeline() -> None:
    import mltgnt

    assert "run_pipeline" not in mltgnt.__all__
    assert not hasattr(mltgnt, "run_pipeline")


def test_interfaces_no_longer_exports_chat_pipeline_protocol() -> None:
    import mltgnt.interfaces as interfaces

    assert "ChatPipelineProtocol" not in interfaces.__all__
    with pytest.raises(ImportError):
        from mltgnt.interfaces.chat import ChatPipelineProtocol  # noqa: F401


def test_changelog_removed_lists_six_modules() -> None:
    text = (_REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "### Removed" in text
    for name in ("loops", "ooda", "improvement", "kpi", "chat", "execution"):
        assert name in text
