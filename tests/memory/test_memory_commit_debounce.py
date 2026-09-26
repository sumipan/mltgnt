"""tests/memory/test_memory_commit_debounce.py — debounced memory commits (nexus #3833).

Uses a throwaway ghdag LocalGitSink repository under tmp_path; no mocks.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import pytest
from ghdag.vcs import LocalGitSink

from mltgnt.config import MemoryConfig
from mltgnt.memory import append_memory_entry, compact, flush_memory_commits, memory_file_path
from mltgnt.memory._format import MemoryEntry, serialize_entry
from mltgnt.memory.dream._format import DreamSection, DreamSummary
from mltgnt.memory.dream.api import write_dream, write_global

_ALLOW = ("chat/memory/", "chat/dream/")


def _git(work: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(work), *args], check=True, capture_output=True, text=True
    ).stdout


def _count(work: Path) -> int:
    return int(_git(work, "rev-list", "--count", "HEAD").strip())


def _subjects(work: Path, n: int) -> list[str]:
    return _git(work, "log", f"-{n}", "--format=%s").splitlines()


def _setup_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    allow_prefixes: tuple[str, ...] = _ALLOW,
    enable_git: bool = True,
) -> Path:
    sink = LocalGitSink.create(tmp_path, owner="mltgnt", allow_prefixes=list(_ALLOW))
    prefixes = ", ".join(f'"{p}"' for p in allow_prefixes)
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
    if enable_git:
        monkeypatch.setenv("ENABLE_GIT", "1")
    return sink.repo_root


def _config(work: Path, debounce: float) -> MemoryConfig:
    return MemoryConfig(chat_dir=work / "chat", commit_debounce_sec=debounce)


def _append(config: MemoryConfig, persona: str, i: int) -> bool:
    return append_memory_entry(
        config,
        persona,
        "user",
        f"hello {i}",
        f"2026-09-26T00:00:0{i}+09:00",
        source_tag="chat",
    )


def _lines(config: MemoryConfig, persona: str) -> int:
    return len(memory_file_path(config, persona).read_text(encoding="utf-8").splitlines())


def test_m1_three_appends_make_one_commit_on_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = _setup_sink(tmp_path, monkeypatch)
    config = _config(work, 300)
    before = _count(work)
    for i in range(3):
        assert _append(config, "p1", i) is True
    assert _count(work) == before
    assert flush_memory_commits() == 1
    assert _count(work) == before + 1
    assert _subjects(work, 1) == ["mltgnt(memory): p1 append"]
    assert "Layer: mltgnt" in _git(work, "log", "-1", "--format=%B")
    assert _lines(config, "p1") == 3


def test_m2_timer_fires_after_debounce(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = _setup_sink(tmp_path, monkeypatch)
    config = _config(work, 0.2)
    before = _count(work)
    assert _append(config, "p1", 0) is True
    deadline = time.monotonic() + 5.0
    while _count(work) == before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _count(work) == before + 1


def test_m3_zero_debounce_commits_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = _setup_sink(tmp_path, monkeypatch)
    config = _config(work, 0)
    before = _count(work)
    assert _append(config, "p1", 0) is True
    assert _count(work) == before + 1


def test_m4_enable_git_unset_makes_no_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = _setup_sink(tmp_path, monkeypatch, enable_git=False)
    config = _config(work, 300)
    before = _count(work)
    for i in range(3):
        assert _append(config, "p1", i) is True
    flush_memory_commits()
    assert _count(work) == before
    assert _lines(config, "p1") == 3


def test_m5_different_personas_commit_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = _setup_sink(tmp_path, monkeypatch)
    config = _config(work, 300)
    before = _count(work)
    assert _append(config, "p1", 0) is True
    assert _append(config, "p2", 1) is True
    assert flush_memory_commits() == 2
    assert _count(work) == before + 2
    assert sorted(_subjects(work, 2)) == ["mltgnt(memory): p1 append", "mltgnt(memory): p2 append"]


def test_m6_commit_failure_is_logged_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    work = _setup_sink(tmp_path, monkeypatch, allow_prefixes=("other/",))
    config = _config(work, 300)
    before = _count(work)
    assert _append(config, "p1", 0) is True
    with caplog.at_level(logging.WARNING, logger="mltgnt.memory._commit"):
        flush_memory_commits()
    assert _lines(config, "p1") == 1
    assert _count(work) == before
    failures = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "memory commit failed" in r.getMessage()
    ]
    assert len(failures) == 1


def _summary(persona: str) -> DreamSummary:
    return DreamSummary(
        persona=persona,
        sections=[DreamSection(category="topics", content="weather", source_entries=1)],
        updated_at="2026-09-26T00:00:00+09:00",
    )


def test_m7_write_dream_and_global_commit_once_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = _setup_sink(tmp_path, monkeypatch)
    before = _count(work)
    write_dream(work / "chat", _summary("p1"), memory_dir_name="dream")
    assert flush_memory_commits() == 1
    assert _count(work) == before + 1
    assert _subjects(work, 1) == ["mltgnt(memory): p1 dream"]

    write_global(work / "chat", _summary("__global__"), memory_dir_name="dream")
    assert flush_memory_commits() == 1
    assert _count(work) == before + 2
    assert _subjects(work, 1) == ["mltgnt(memory): __global__ global"]


def test_m8_compact_commits_only_on_real_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = _setup_sink(tmp_path, monkeypatch)
    config = _config(work, 300)
    path = memory_file_path(config, "p1")
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = MemoryEntry(
        timestamp="2026-09-20T00:00:00+09:00",
        role="user",
        content="hello world",
        source_tag="chat",
        layer="recent",
    )
    path.write_text(serialize_entry(entry) + "\n", encoding="utf-8")
    before = _count(work)

    compact(config, "p1", llm_call=lambda prompt: "ok", dry_run=True)
    assert flush_memory_commits() == 0
    assert _count(work) == before

    compact(config, "p1", llm_call=lambda prompt: "ok")
    assert flush_memory_commits() == 1
    assert _count(work) == before + 1
    assert _subjects(work, 1) == ["mltgnt(memory): p1 compact"]
