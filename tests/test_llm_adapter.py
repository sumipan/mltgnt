"""Integration tests for mltgnt.bridges.llm_adapter.call_llm.

Refs: sumipan/nexus#6683
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from mltgnt.bridges.llm_adapter import call_llm

_CLAUDE_SUCCESS_LINE = (
    '{"type":"result","subtype":"success","is_error":false,"result":"pong"}'
)


def _model_from_argv(argv_file: Path) -> str:
    lines = argv_file.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line == "--model" and index + 1 < len(lines):
            return lines[index + 1]
    raise AssertionError("--model not found in captured argv")


def _install_fake_claude(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_file = tmp_path / "argv.txt"
    script = bin_dir / "claude"
    script.write_text(
        f"""#!/bin/sh
argv_file="{argv_file}"
for arg in "$@"; do
  echo "$arg" >> "$argv_file"
done
cat > /dev/null
printf '%s\\n' '{_CLAUDE_SUCCESS_LINE}'
exit 0
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return argv_file


def _wire_fake_claude_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    _install_fake_claude(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("GHDAG_LLM_MODELS", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path / "argv.txt"


def test_call_llm_passes_model_to_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv_file = _wire_fake_claude_env(tmp_path, monkeypatch)

    result = call_llm(
        "ping",
        engine="claude",
        model="claude-haiku-4-5-20251001",
    )

    assert result.ok is True
    assert result.body == "pong"
    assert result.engine_used == "claude"
    assert _model_from_argv(argv_file) == "claude-haiku-4-5-20251001"


def test_call_llm_empty_model_uses_engine_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv_file = _wire_fake_claude_env(tmp_path, monkeypatch)

    result = call_llm("ping", engine="claude", model="")

    assert result.ok is True
    assert result.body == "pong"
    assert _model_from_argv(argv_file) == "claude-sonnet-4-6"
