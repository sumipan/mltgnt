"""Tests for mltgnt memory dream show/forget CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mltgnt.memory.dream import DreamSection, DreamSummary, read_dream, write_dream


def _run_mltgnt(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "mltgnt", *args]
    return subprocess.run(
        cmd,
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )


def _worktree_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_dream(persona_dir: Path, sections: list[DreamSection]) -> None:
    summary = DreamSummary(
        persona=persona_dir.name,
        sections=sections,
        updated_at="2026-06-07T12:00:00+09:00",
    )
    write_dream(persona_dir, summary)


@pytest.fixture
def persona_setup(tmp_path: Path) -> tuple[Path, str]:
    chat_dir = tmp_path / "agents"
    chat_dir.mkdir()
    persona = "alice"
    persona_dir = chat_dir / persona
    persona_dir.mkdir()
    return chat_dir, persona


def test_dream_show_outputs_sections(persona_setup: tuple[Path, str]) -> None:
    chat_dir, persona = persona_setup
    _write_dream(
        chat_dir / persona,
        [
            DreamSection(category="behavior patterns", content="morning type", source_entries=2),
            DreamSection(category="preferences", content="concise", source_entries=1),
        ],
    )

    result = _run_mltgnt(
        "memory", "dream", "show", persona,
        "--chat-dir", str(chat_dir),
        cwd=_worktree_root(),
    )

    assert result.returncode == 0
    assert "behavior patterns" in result.stdout
    assert "morning type" in result.stdout
    assert "source_entries: 2" in result.stdout
    assert "preferences" in result.stdout
    assert "concise" in result.stdout
    assert "source_entries: 1" in result.stdout


def test_dream_show_no_dream_exits_zero(persona_setup: tuple[Path, str]) -> None:
    chat_dir, persona = persona_setup

    result = _run_mltgnt(
        "memory", "dream", "show", persona,
        "--chat-dir", str(chat_dir),
        cwd=_worktree_root(),
    )

    assert result.returncode == 0
    assert f"No dream summary found for {persona}" in result.stdout


def test_dream_forget_removes_category(persona_setup: tuple[Path, str]) -> None:
    chat_dir, persona = persona_setup
    persona_dir = chat_dir / persona
    _write_dream(
        persona_dir,
        [
            DreamSection(category="behavior patterns", content="to delete", source_entries=1),
            DreamSection(category="preferences", content="keep", source_entries=1),
        ],
    )

    result = _run_mltgnt(
        "memory", "dream", "forget", persona,
        "--category", "behavior patterns",
        "--chat-dir", str(chat_dir),
        cwd=_worktree_root(),
    )

    assert result.returncode == 0
    loaded = read_dream(persona_dir)
    assert loaded is not None
    assert len(loaded.sections) == 1
    assert loaded.sections[0].category == "preferences"
    assert loaded.sections[0].content == "keep"


def test_dream_forget_no_dream_exits_one(persona_setup: tuple[Path, str]) -> None:
    chat_dir, persona = persona_setup

    result = _run_mltgnt(
        "memory", "dream", "forget", persona,
        "--category", "behavior patterns",
        "--chat-dir", str(chat_dir),
        cwd=_worktree_root(),
    )

    assert result.returncode == 1
    assert "No dream summary found" in result.stderr


def test_dream_forget_missing_category_exits_one(persona_setup: tuple[Path, str]) -> None:
    chat_dir, persona = persona_setup
    _write_dream(
        chat_dir / persona,
        [DreamSection(category="behavior patterns", content="text", source_entries=1)],
    )

    result = _run_mltgnt(
        "memory", "dream", "forget", persona,
        "--category", "nonexistent category",
        "--chat-dir", str(chat_dir),
        cwd=_worktree_root(),
    )

    assert result.returncode == 1
    assert "Category not found" in result.stderr
