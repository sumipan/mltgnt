"""Tests for mltgnt.memory.reflection (Issue #4037)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from mltgnt.memory._format import MemoryEntry
from mltgnt.memory.reflection import (
    ReflectionAdd,
    ReflectionParseError,
    ReflectionResult,
    apply_reflection,
    build_reflection_prompt,
    parse_reflection,
)
from mltgnt.memory.semantic import SemanticStore

TS = "2026-09-26 03:00"


def _store(tmp_path: Path) -> SemanticStore:
    return SemanticStore(tmp_path / "semantic.jsonl", persona_stem="p")


def test_build_prompt_includes_inputs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    active = [store.append("fact", "likes tea", "user", "s", ts=TS)]
    episodes = [MemoryEntry(timestamp="2026-09-25 10:00", role="user", content="hello\nthere", source_tag="chat")]
    prompt = build_reflection_prompt(episodes, active, persona="helper")
    assert "You are helper." in prompt
    assert "2026-09-25 10:00 user: hello there" in prompt
    assert "m_2026-09-26_0001 [fact] (user) likes tea" in prompt
    assert '"supersede"' in prompt
    assert prompt.isascii()


def test_build_prompt_empty_inputs() -> None:
    assert "(none)" in build_reflection_prompt([], [], persona="helper")


def test_parse_reflection_valid_with_fence() -> None:
    text = (
        "```json\n"
        + json.dumps(
            {
                "reflection": "learned things",
                "add": [{"kind": "fact", "content": "has a dog", "subject": "user"}],
                "supersede": ["m_1"],
            }
        )
        + "\n```"
    )
    result = parse_reflection(text)
    assert result == ReflectionResult(
        reflection="learned things",
        add=(ReflectionAdd(kind="fact", content="has a dog", subject="user"),),
        supersede=("m_1",),
    )


@pytest.mark.parametrize(
    "text",
    [
        "not json at all",
        "{broken",
        "[1, 2]",
        '{"reflection": 1}',
        '{"add": "x"}',
        '{"supersede": [1]}',
    ],
)
def test_parse_reflection_rejects_malformed(text: str) -> None:
    with pytest.raises(ReflectionParseError):
        parse_reflection(text)


def test_apply_truncates_adds_to_max(tmp_path: Path) -> None:
    store = _store(tmp_path)
    adds = tuple(ReflectionAdd(kind="fact", content=f"fact {i}", subject="user") for i in range(7))
    report = apply_reflection(store, ReflectionResult(reflection="", add=adds), run_id="r1", ts=TS)
    assert len(report.added) == 5
    assert report.truncated == 2
    assert [e.content for e in store.active()] == [f"fact {i}" for i in range(5)]
    assert all(e.source == "reflection:r1" for e in store.active())


def test_apply_validates_and_dedupes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append("fact", "likes tea", "user", "s", ts=TS)
    adds = (
        ReflectionAdd(kind="fact", content="Likes tea", subject="user"),
        ReflectionAdd(kind="unknown", content="x", subject="user"),
        ReflectionAdd(kind="fact", content="y", subject="team"),
        ReflectionAdd(kind="self", content="i ramble", subject="self"),
    )
    report = apply_reflection(store, ReflectionResult(reflection="", add=adds), run_id="r1", ts=TS)
    assert report.rejected == [("Likes tea", "duplicate"), ("x", "invalid_kind"), ("y", "invalid_subject")]
    assert len(report.added) == 1


def test_apply_supersede_and_reflection(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    store = _store(tmp_path)
    old = store.append("fact", "lives in paris", "user", "s", ts=TS)
    gone = store.append("fact", "old", "user", "s", ts=TS)
    store.supersede(gone.id)
    result = ReflectionResult(
        reflection="user moved",
        add=(ReflectionAdd(kind="fact", content="lives in lyon", subject="user"),),
        supersede=(old.id, gone.id, "m_missing"),
    )
    with caplog.at_level(logging.WARNING, logger="mltgnt.memory.reflection"):
        report = apply_reflection(store, result, run_id="r2", ts=TS)
    assert report.superseded == [old.id]
    assert report.ignored_supersede == [gone.id, "m_missing"]
    assert sum("ignoring supersede" in r.getMessage() for r in caplog.records) == 2
    assert report.reflection_id is not None
    reflection = store.get(report.reflection_id)
    assert reflection is not None
    assert (reflection.kind, reflection.subject, reflection.content) == ("reflection", "self", "user moved")
    assert {e.content for e in store.active()} == {"lives in lyon", "user moved"}
