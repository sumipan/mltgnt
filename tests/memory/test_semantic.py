"""Tests for mltgnt.memory.semantic (Issue #4037)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mltgnt.memory.semantic import KINDS, SemanticEntry, SemanticStore, validate_entry


def _store(tmp_path: Path) -> SemanticStore:
    return SemanticStore(tmp_path / "p" / "semantic.jsonl", persona_stem="p")


def _entry(**overrides: str) -> SemanticEntry:
    base = dict(id="m_x", ts="2026-09-26 14:05", kind="fact", content="likes tea", subject="user", source="s")
    base.update(overrides)
    return SemanticEntry(**base)


def test_kinds_vocabulary() -> None:
    assert KINDS == ("fact", "preference", "commitment", "caveat", "self", "reflection")


@pytest.mark.parametrize(
    "subject",
    ["user", "self", "unresolved", "person:alice", "project:nexus", "skill:review"],
)
def test_validate_entry_accepts_subject_forms(subject: str) -> None:
    assert validate_entry(_entry(subject=subject)) == []


@pytest.mark.parametrize("subject", ["team", "", "person:", "project: x", "User"])
def test_validate_entry_rejects_bad_subject(subject: str) -> None:
    assert "invalid_subject" in validate_entry(_entry(subject=subject))


def test_validate_entry_lists_all_violations() -> None:
    errors = validate_entry(_entry(kind="unknown", subject="team", content="  "))
    assert errors == ["invalid_kind", "invalid_subject", "empty_content"]


def test_append_writes_one_line_with_sequential_ids(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = store.append("fact", "likes tea", "user", "slack:c:1", ts="2026-09-26 14:05")
    b = store.append("caveat", "no calls", "user", "slack:c:2", ts="2026-09-26 15:00")
    c = store.append("fact", "has a cat", "user", "slack:c:3", ts="2026-09-27 09:00")
    assert (a.id, b.id, c.id) == ("m_2026-09-26_0001", "m_2026-09-26_0002", "m_2026-09-27_0001")
    lines = store.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first == {
        "id": "m_2026-09-26_0001",
        "ts": "2026-09-26 14:05",
        "kind": "fact",
        "content": "likes tea",
        "subject": "user",
        "source": "slack:c:1",
        "status": "active",
    }


def test_append_rejects_invalid_entry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="invalid_kind"):
        store.append("unknown", "x", "user", "s", ts="2026-09-26 14:05")
    with pytest.raises(ValueError, match="invalid_subject"):
        store.append("fact", "x", "team", "s", ts="2026-09-26 14:05")
    assert not store.path.exists()


def test_supersede_marks_status_and_keeps_line(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = store.append("fact", "likes tea", "user", "s", ts="2026-09-26 14:05")
    b = store.append("fact", "likes coffee", "user", "s", ts="2026-09-26 14:06", supersedes=a.id)
    assert store.supersede(a.id) is True
    assert store.supersede(a.id) is False
    assert store.supersede("m_missing") is False
    assert [e.id for e in store.active()] == [b.id]
    got = store.get(a.id)
    assert got is not None and got.status == "superseded"
    assert store.get(b.id).supersedes == a.id  # type: ignore[union-attr]
    assert len(store.path.read_text(encoding="utf-8").splitlines()) == 2
    assert not store.path.with_name("semantic.jsonl.tmp").exists()


def test_find_duplicate_normalizes_whitespace_and_case(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = store.append("fact", "Likes  green\ttea", "user", "s", ts="2026-09-26 14:05")
    assert store.find_duplicate("likes green tea", "fact", "user") == a
    assert store.find_duplicate("likes green tea", "preference", "user") is None
    assert store.find_duplicate("likes green tea", "fact", "self") is None
    store.supersede(a.id)
    assert store.find_duplicate("likes green tea", "fact", "user") is None


def test_broken_lines_are_skipped_and_preserved(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = store.append("fact", "likes tea", "user", "s", ts="2026-09-26 14:05")
    with store.path.open("a", encoding="utf-8") as f:
        f.write("{broken\n")
    assert [e.id for e in store.entries()] == [a.id]
    store.supersede(a.id)
    assert "{broken" in store.path.read_text(encoding="utf-8")


def test_missing_file_is_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.entries() == []
    assert store.get("m_x") is None
