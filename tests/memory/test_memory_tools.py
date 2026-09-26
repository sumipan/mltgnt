"""Tests for mltgnt.memory.tools (Issue #4037)."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from mltgnt.memory._format import MemoryEntry, serialize_entry
from mltgnt.memory.core_render import render_core
from mltgnt.memory.semantic import SemanticStore
from mltgnt.memory.tools import (
    MEMORY_TOOL_NAMES,
    MEMORY_TOOL_SPECS,
    MemoryGate,
    MemoryToolExecutor,
    query_terms,
)

_LOGGER = "mltgnt.memory.tools"


def _executor(tmp_path: Path, gate: MemoryGate | None = None, **kwargs: Any) -> MemoryToolExecutor:
    persona_dir = tmp_path / "p"
    store = SemanticStore(persona_dir / "semantic.jsonl", persona_stem="p")
    return MemoryToolExecutor(
        store,
        episodes_path=persona_dir / "episodes.jsonl",
        gate=gate or MemoryGate(),
        source="slack:c1:100.1",
        now=lambda: datetime(2026, 9, 26, 14, 5),
        **kwargs,
    )


def _call(ex: MemoryToolExecutor, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return json.loads(ex(name, args))


def _lines(ex: MemoryToolExecutor) -> int:
    if not ex.store.path.exists():
        return 0
    return len(ex.store.path.read_text(encoding="utf-8").splitlines())


def _remember(ex: MemoryToolExecutor, content: str, kind: str = "fact", subject: str = "user") -> dict[str, Any]:
    return _call(ex, "remember", {"content": content, "kind": kind, "subject": subject})


def test_tool_specs_shape() -> None:
    assert {"remember", "recall", "forget"} == MEMORY_TOOL_NAMES
    for spec in MEMORY_TOOL_SPECS:
        assert spec["description"]
        assert spec["parameters"]["type"] == "object"
    remember = next(s for s in MEMORY_TOOL_SPECS if s["name"] == "remember")
    assert remember["parameters"]["required"] == ["content", "kind", "subject"]


def test_remember_then_recall(tmp_path: Path) -> None:
    ex = _executor(tmp_path)
    res = _remember(ex, "user prefers green tea in the morning", kind="preference")
    assert res == {"ok": True, "id": "m_2026-09-26_0001"}
    assert _lines(ex) == 1
    entry = ex.store.get("m_2026-09-26_0001")
    assert entry is not None
    assert entry.source == "slack:c1:100.1"
    assert entry.ts == "2026-09-26 14:05"
    out = ex("recall", {"query": "green tea"})
    assert "user prefers green tea in the morning" in out
    assert "m_2026-09-26_0001" in out


def test_source_callable(tmp_path: Path) -> None:
    ex = _executor(tmp_path)
    ex._source = lambda: "slack:c9:9.9"
    _remember(ex, "a fact")
    assert ex.store.active()[0].source == "slack:c9:9.9"


def _assert_rejected(
    ex: MemoryToolExecutor,
    caplog: pytest.LogCaptureFixture,
    args: dict[str, Any],
    reason: str,
) -> None:
    before = _lines(ex)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        res = _call(ex, "remember", args)
    assert res == {"ok": False, "reason": reason}
    assert _lines(ex) == before
    warnings = [r for r in caplog.records if r.name == _LOGGER and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert reason in warnings[0].getMessage()


def test_gate_per_turn_limit_and_reset(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ex = _executor(tmp_path)
    for i in range(3):
        assert _remember(ex, f"fact number {i}")["ok"] is True
    _assert_rejected(ex, caplog, {"content": "fact number 4", "kind": "fact", "subject": "user"}, "per_turn_limit")
    ex.reset_turn()
    assert _remember(ex, "fact number 4")["ok"] is True


def test_gate_too_long(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ex = _executor(tmp_path)
    assert _remember(ex, "a" * 300)["ok"] is True
    _assert_rejected(ex, caplog, {"content": "b" * 301, "kind": "fact", "subject": "user"}, "too_long")


def test_gate_secret(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ex = _executor(tmp_path, MemoryGate(secret_check=lambda text: "token=" in text))
    _assert_rejected(ex, caplog, {"content": "token=abc123", "kind": "fact", "subject": "user"}, "secret")
    warning_text = caplog.records[-1].getMessage()
    assert "abc123" not in warning_text


def test_gate_duplicate(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ex = _executor(tmp_path)
    _remember(ex, "likes green tea")
    _assert_rejected(ex, caplog, {"content": "Likes  green tea", "kind": "fact", "subject": "user"}, "duplicate")


def test_gate_subject_not_allowed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ex = _executor(tmp_path, MemoryGate(allowed_subject_prefixes=("self", "project:")))
    _assert_rejected(ex, caplog, {"content": "x", "kind": "fact", "subject": "user"}, "subject_not_allowed")
    assert _remember(ex, "x", subject="self")["ok"] is True
    assert _remember(ex, "y", subject="project:nexus")["ok"] is True


def test_invalid_kind_and_subject(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ex = _executor(tmp_path)
    _assert_rejected(ex, caplog, {"content": "x", "kind": "unknown", "subject": "user"}, "invalid_kind")
    _assert_rejected(ex, caplog, {"content": "x", "kind": "fact", "subject": "team"}, "invalid_subject")


def test_subject_resolver(tmp_path: Path) -> None:
    ex = _executor(tmp_path, subject_resolver=lambda s: "user" if s == "me" else s)
    assert _remember(ex, "x", subject="me")["ok"] is True
    assert ex.store.active()[0].subject == "user"


def test_forget_by_id_hides_from_core(tmp_path: Path) -> None:
    ex = _executor(tmp_path)
    mid = _remember(ex, "never call after nine", kind="caveat")["id"]
    assert "never call after nine" in render_core(ex.store.entries())
    assert _call(ex, "forget", {"id": mid}) == {"ok": True, "id": mid}
    assert ex.store.get(mid).status == "superseded"  # type: ignore[union-attr]
    assert "never call after nine" not in render_core(ex.store.entries())
    assert _call(ex, "forget", {"id": mid}) == {"ok": False, "reason": "not_found"}


def test_forget_by_text(tmp_path: Path) -> None:
    ex = _executor(tmp_path)
    a = _remember(ex, "likes green tea")["id"]
    b = _remember(ex, "likes black tea")["id"]
    res = _call(ex, "forget", {"text": "tea"})
    assert res["ok"] is False
    assert res["reason"] == "ambiguous"
    assert sorted(res["candidates"]) == sorted([a, b])
    assert {e.id for e in ex.store.active()} == {a, b}
    assert _call(ex, "forget", {"text": "GREEN"}) == {"ok": True, "id": a}
    assert _call(ex, "forget", {"text": "coffee"}) == {"ok": False, "reason": "not_found"}
    assert _call(ex, "forget", {}) == {"ok": False, "reason": "invalid_args"}


def test_recall_filters_and_ranks(tmp_path: Path) -> None:
    ex = _executor(tmp_path)
    _remember(ex, "project alpha deadline friday", kind="commitment", subject="project:alpha")
    _remember(ex, "alpha team likes pizza", kind="fact", subject="user")
    _remember(ex, "unrelated note", kind="fact", subject="user")
    episode = MemoryEntry(timestamp="2026-09-25 10:00", role="user", content="talked about alpha", source_tag="chat")
    ex.episodes_path.write_text(serialize_entry(episode) + "\n", encoding="utf-8")

    out = ex("recall", {"query": "alpha deadline"}).splitlines()
    assert out[0].endswith("project alpha deadline friday")
    assert any("[episode 2026-09-25 10:00] user: talked about alpha" in line for line in out)
    assert not any("unrelated" in line for line in out)

    only_fact = ex("recall", {"query": "alpha", "kind": "fact"})
    assert only_fact.splitlines() == ["[m_2026-09-26_0002] [fact] (user) alpha team likes pizza"]

    by_subject = ex("recall", {"query": "alpha", "subject": "project:alpha"})
    assert "deadline" in by_subject and "pizza" not in by_subject and "episode" not in by_subject

    assert len(ex("recall", {"query": "alpha", "limit": 1}).splitlines()) == 1
    assert ex("recall", {"query": "zebra"}) == "No matching memories."


def test_recall_limit_is_capped(tmp_path: Path) -> None:
    ex = _executor(tmp_path, MemoryGate(max_per_turn=100))
    for i in range(25):
        _remember(ex, f"item {i} common")
    assert len(ex("recall", {"query": "common", "limit": 50}).splitlines()) == 20
    assert len(ex("recall", {"query": "common"}).splitlines()) == 10


def test_query_terms_bigrams_for_non_ascii() -> None:
    assert query_terms("Tea  time") == ["tea", "time"]
    assert query_terms("cafés") == ["ca", "af", "fé", "és"]


def test_executor_never_raises(tmp_path: Path) -> None:
    ex = _executor(tmp_path)
    assert json.loads(ex("unknown_tool", {})) == {"ok": False, "reason": "unknown_tool"}

    def boom(_: str) -> bool:
        raise RuntimeError("boom")

    ex.gate = MemoryGate(secret_check=boom)
    res = _call(ex, "remember", {"content": "x", "kind": "fact", "subject": "user"})
    assert res["ok"] is False and res["reason"] == "error"
