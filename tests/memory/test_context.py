"""Tests for mltgnt.memory.context (Issue #6238)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import dataclasses

from mltgnt.config.language import EN
from mltgnt.memory import MemoryEntry, serialize_entry
from mltgnt.memory.context import (
    build_memory_context,
    recent_episodes_text,
    thread_tail_text,
)
from mltgnt.memory.semantic import SemanticStore
from mltgnt.persona.memory import LEGACY_DIRECT_DEDUPE_PREFIX


def _episode(dedupe_key: str, content: str, ts: str = "2026-09-27 10:00") -> MemoryEntry:
    return MemoryEntry(
        timestamp=ts,
        role="user",
        content=content,
        source_tag="[slack]",
        dedupe_key=dedupe_key,
    )


def _write_episodes(path: Path, entries: list[MemoryEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(serialize_entry(e) + "\n" for e in entries), encoding="utf-8")


def test_thread_tail_filters_by_full_thread_token(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    lines = [
        _episode(f"{LEGACY_DIRECT_DEDUPE_PREFIX}:S:T:1:user", "thread ST"),
        _episode(f"{LEGACY_DIRECT_DEDUPE_PREFIX}:S:T2:2:user", "thread ST2"),
    ]
    _write_episodes(path, lines)
    out = thread_tail_text(path, space_id="S", thread_key="T", max_entries=5)
    assert "thread ST" in out
    assert "thread ST2" not in out


def test_thread_tail_applies_limit_and_empty_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    base = f"{LEGACY_DIRECT_DEDUPE_PREFIX}:S:T"
    _write_episodes(
        path,
        [_episode(f"{base}:{i}:user", f"msg{i}") for i in range(5)],
    )
    limited = thread_tail_text(path, space_id="S", thread_key="T", max_entries=2)
    assert "msg3" in limited and "msg4" in limited
    assert "msg0" not in limited
    missing = tmp_path / "missing.jsonl"
    assert thread_tail_text(missing, space_id="S", thread_key="T", max_entries=5) == ""
    assert thread_tail_text(path, space_id="S", thread_key="", max_entries=5) == ""
    assert thread_tail_text(path, space_id="S", thread_key="T", max_entries=0) == ""


def test_recent_episodes_filters_since_and_thread(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    since = datetime(2026, 9, 27, 10, 0)
    entries = [
        _episode(f"{LEGACY_DIRECT_DEDUPE_PREFIX}:S:A:1:user", "old other", "2026-09-27 09:59"),
        _episode(f"{LEGACY_DIRECT_DEDUPE_PREFIX}:S:B:2:user", "keep other", "2026-09-27 10:00"),
        _episode(f"{LEGACY_DIRECT_DEDUPE_PREFIX}:S:A:3:user", "exclude same", "2026-09-27 10:05"),
        _episode(f"{LEGACY_DIRECT_DEDUPE_PREFIX}:S:B:4:user", "keep other 2", "2026-09-27 10:10"),
    ]
    _write_episodes(path, entries)
    out = recent_episodes_text(
        path,
        since=since,
        max_entries=10,
        exclude_thread_key="A",
    )
    assert "keep other" in out and "keep other 2" in out
    assert "old other" not in out
    assert "exclude same" not in out


def test_recent_episodes_skips_invalid_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    _write_episodes(
        path,
        [
            _episode(f"{LEGACY_DIRECT_DEDUPE_PREFIX}:S:B:1:user", "bad ts", "not-a-date"),
            _episode(f"{LEGACY_DIRECT_DEDUPE_PREFIX}:S:B:2:user", "good", "2026-09-27 10:01"),
        ],
    )
    out = recent_episodes_text(
        path,
        since=datetime(2026, 9, 27, 10, 0),
        max_entries=5,
    )
    assert "good" in out
    assert "bad ts" not in out


def test_build_memory_context_composes_enabled_sections(tmp_path: Path) -> None:
    persona = tmp_path / "p"
    persona.mkdir()
    store = SemanticStore(persona / "semantic.jsonl", persona_stem="p")
    store.append("fact", "core fact", "user", "test", ts="2026-09-27 10:00")
    ep = persona / "episodes.jsonl"
    _write_episodes(
        ep,
        [
            _episode(f"{LEGACY_DIRECT_DEDUPE_PREFIX}:C:T:1:user", "same thread", "2026-09-27 11:00"),
            _episode(f"{LEGACY_DIRECT_DEDUPE_PREFIX}:C:U:2:user", "other thread", "2026-09-27 11:30"),
        ],
    )
    text = build_memory_context(
        persona,
        persona_stem="p",
        space_id="C",
        thread_key="T",
        recent_hours=24,
        recent_entries=5,
        now=datetime(2026, 9, 27, 12, 0),
    )
    assert "core fact" in text
    assert "same thread" in text
    assert "other thread" in text
    assert "\n\n" in text


def test_build_memory_context_uses_heading_precedence(tmp_path: Path) -> None:
    persona = tmp_path / "p"
    persona.mkdir()
    store = SemanticStore(persona / "semantic.jsonl", persona_stem="p")
    store.append("fact", "fact body", "user", "test", ts="2026-09-27 10:00")
    pack = dataclasses.replace(
        EN,
        memory_core_heading="PACK CORE",
        memory_thread_tail_heading="PACK THREAD",
        memory_recent_heading="PACK RECENT",
    )
    text_pack = build_memory_context(
        persona,
        persona_stem="p",
        space_id=None,
        thread_key=None,
        pack=pack,
    )
    assert "PACK CORE" in text_pack

    text_explicit = build_memory_context(
        persona,
        persona_stem="p",
        space_id=None,
        thread_key=None,
        pack=pack,
        core_heading="EXPLICIT CORE",
    )
    assert "EXPLICIT CORE" in text_explicit
    assert "PACK CORE" not in text_explicit

    text_default = build_memory_context(
        persona,
        persona_stem="p",
        space_id=None,
        thread_key=None,
        pack=None,
    )
    assert "## Memory (core)" in text_default


def test_build_memory_context_empty_inputs_return_empty(tmp_path: Path) -> None:
    persona = tmp_path / "empty"
    persona.mkdir()
    assert (
        build_memory_context(
            persona,
            persona_stem="empty",
            space_id="C",
            thread_key=None,
        )
        == ""
    )
    assert (
        build_memory_context(
            persona,
            persona_stem="empty",
            space_id="C",
            thread_key="T",
        )
        == ""
    )
