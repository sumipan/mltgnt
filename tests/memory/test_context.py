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


def _episode(
    dedupe_key: str,
    content: str,
    ts: str = "2026-09-27 10:00",
    *,
    peer: str | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        timestamp=ts,
        role="user",
        content=content,
        source_tag="[slack]",
        dedupe_key=dedupe_key,
        peer=peer,
    )


def _direct_key(space: str, thread: str, message_id: str, role: str = "user") -> str:
    return f"direct:{space}:{thread}:{message_id}:{role}"


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


def test_recent_episodes_filters_by_peer(tmp_path: Path) -> None:
    """Issue #7324: peer filter keeps only matching episodic lines."""
    path = tmp_path / "episodes.jsonl"
    _write_episodes(
        path,
        [
            _episode(_direct_key("S", "thread-a", "1"), "alice one", peer="alice"),
            _episode(_direct_key("S", "thread-a", "2"), "alice two", peer="alice"),
            _episode(_direct_key("S", "thread-b", "3"), "bob one", peer="bob"),
            _episode(_direct_key("S", "thread-b", "4"), "bob two", peer="bob"),
            _episode(_direct_key("S", "thread-c", "5"), "legacy line"),
        ],
    )
    out = recent_episodes_text(
        path,
        since=datetime(2026, 9, 27, 9, 0),
        max_entries=20,
        peer="alice",
    )
    assert "alice one" in out and "alice two" in out
    assert "bob one" not in out and "bob two" not in out
    assert "legacy line" not in out


def test_recent_episodes_caps_each_thread_before_the_total(tmp_path: Path) -> None:
    """Issue #7324: per-thread cap runs before the global tail limit."""
    path = tmp_path / "episodes.jsonl"
    entries = []
    for i in range(6):
        entries.append(
            _episode(
                _direct_key("S", "thread-x", str(i)),
                f"x-old-{i}",
                ts=f"2026-09-27 09:{i:02d}",
            ),
        )
    for i in range(6):
        entries.append(
            _episode(
                _direct_key("S", "thread-y", str(i)),
                f"y-new-{i}",
                ts=f"2026-09-27 11:{i:02d}",
            ),
        )
    _write_episodes(path, entries)
    since = datetime(2026, 9, 27, 8, 0)
    uncapped = recent_episodes_text(path, since=since, max_entries=6)
    assert "y-new-5" in uncapped and "x-old-0" not in uncapped
    capped = recent_episodes_text(
        path,
        since=since,
        max_entries=6,
        per_thread_entries=2,
    )
    assert capped.splitlines()
    for i in range(4):
        assert f"x-old-{i}" not in capped and f"y-new-{i}" not in capped
    assert "x-old-4" in capped and "x-old-5" in capped
    assert "y-new-4" in capped and "y-new-5" in capped
    x_idx = capped.index("x-old-4")
    y_idx = capped.index("y-new-4")
    assert x_idx < y_idx


def test_recent_episodes_defaults_keep_current_behavior(tmp_path: Path) -> None:
    """Issue #7324: omitting peer and per-thread caps preserves prior selection."""
    path = tmp_path / "episodes.jsonl"
    _write_episodes(
        path,
        [
            _episode(_direct_key("S", "thread-a", "1"), "with peer", peer="alice"),
            _episode(_direct_key("S", "thread-b", "2"), "without peer"),
        ],
    )
    out = recent_episodes_text(
        path,
        since=datetime(2026, 9, 27, 9, 0),
        max_entries=10,
    )
    assert "with peer" in out and "without peer" in out


def test_build_memory_context_passes_recent_peer_and_per_thread(tmp_path: Path) -> None:
    """Issue #7324: build_memory_context forwards peer and per-thread limits."""
    persona = tmp_path / "p"
    persona.mkdir()
    ep = persona / "episodes.jsonl"
    entries = []
    for i in range(3):
        entries.append(
            _episode(
                _direct_key("C", "thread-other", f"a{i}"),
                f"alice-other-{i}",
                ts=f"2026-09-27 10:{i:02d}",
                peer="alice",
            ),
        )
    entries.append(
        _episode(
            _direct_key("C", "thread-other", "b0"),
            "bob-other",
            ts="2026-09-27 10:30",
            peer="bob",
        ),
    )
    entries.append(
        _episode(
            _direct_key("C", "thread-current", "c0"),
            "current-thread",
            ts="2026-09-27 11:00",
            peer="alice",
        ),
    )
    _write_episodes(ep, entries)
    text = build_memory_context(
        persona,
        persona_stem="p",
        space_id="C",
        thread_key="thread-current",
        thread_tail_entries=0,
        recent_hours=48,
        recent_entries=12,
        recent_peer="alice",
        recent_per_thread_entries=2,
        now=datetime(2026, 9, 27, 12, 0),
    )
    assert "alice-other-1" in text and "alice-other-2" in text
    assert "alice-other-0" not in text
    assert "bob-other" not in text
    assert "current-thread" not in text


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
