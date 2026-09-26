"""tests/test_files_adapter_commit.py — mltgnt.bridges.files_adapter.commit (nexus #3833).

Uses a throwaway ghdag LocalGitSink repository under tmp_path; no mocks.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from ghdag.vcs import LocalGitSink, OwnershipError

from mltgnt.bridges.files_adapter import commit

_ALLOW = ("chat/memory/", "chat/dream/")


def _git(work: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(work), *args], check=True, capture_output=True, text=True
    ).stdout


def _count(work: Path) -> int:
    return int(_git(work, "rev-list", "--count", "HEAD").strip())


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sink = LocalGitSink.create(tmp_path, owner="mltgnt", allow_prefixes=list(_ALLOW))
    prefixes = ", ".join(f'"{p}"' for p in _ALLOW)
    cfg = tmp_path / "vcs.yml"
    cfg.write_text(
        f'audit_path: "{tmp_path / "audit.jsonl"}"\n'
        "sinks:\n"
        "  memory:\n"
        f'    repo_root: "{sink.repo_root}"\n'
        "    branch: main\n"
        "    owner: mltgnt\n"
        f"    allow_prefixes: [{prefixes}]\n"
        "    push: immediate\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GHDAG_VCS_CONFIG", str(cfg))
    monkeypatch.setenv("ENABLE_GIT", "1")
    return sink.repo_root


def _write_memory(work: Path) -> Path:
    path = work / "chat" / "memory" / "p1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"role": "user"}\n', encoding="utf-8")
    return path


def _assert_committed(work: Path, result, before: int) -> None:
    assert result.committed is True
    assert result.sha is not None
    assert _count(work) == before + 1
    assert "Layer: mltgnt" in _git(work, "log", "-1", "--format=%B")
    changed = _git(work, "show", "--name-only", "--format=", "HEAD").split()
    assert changed == ["chat/memory/p1.jsonl"]


def test_f1_absolute_path_is_made_relative(work: Path) -> None:
    path = _write_memory(work)
    before = _count(work)
    result = commit([path.resolve()], "mltgnt(memory): p1 append")
    _assert_committed(work, result, before)


def test_f2_relative_path(work: Path) -> None:
    _write_memory(work)
    before = _count(work)
    result = commit(["chat/memory/p1.jsonl"], "mltgnt(memory): p1 append")
    _assert_committed(work, result, before)


def test_f3_enable_git_unset_is_skipped(work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENABLE_GIT", raising=False)
    path = _write_memory(work)
    before = _count(work)
    result = commit([path], "mltgnt(memory): p1 append")
    assert result.skipped is True
    assert result.reason == "ENABLE_GIT unset"
    assert _count(work) == before


def test_f4_path_outside_repo_root_raises_value_error(work: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.jsonl"
    outside.write_text("x\n", encoding="utf-8")
    before = _count(work)
    with pytest.raises(ValueError):
        commit([outside], "mltgnt(memory): p1 append")
    assert _count(work) == before


def test_f5_path_outside_allow_prefixes_raises_ownership_error(work: Path) -> None:
    other = work / "other" / "x.jsonl"
    other.parent.mkdir(parents=True)
    other.write_text("x\n", encoding="utf-8")
    before = _count(work)
    with pytest.raises(OwnershipError):
        commit(["other/x.jsonl"], "mltgnt(memory): p1 append")
    assert _count(work) == before
