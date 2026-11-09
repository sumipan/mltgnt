"""Tests for mltgnt.memory.reflect_job (Issue #6239)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mltgnt.memory._format import serialize_entry, MemoryEntry
from mltgnt.memory.reflect_job import (
    LLM_VERSION,
    apply_from_payloads,
    build_collect_payload,
    collect_new_episodes,
    load_reflect_state,
    save_reflect_state,
    validate_llm_payload,
)
from mltgnt.memory.semantic import SemanticStore

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
T1 = "2026-01-10 09:00"
T2 = "2026-01-12 10:00"
T3 = "2026-01-14 11:00"


def _episode(ts: str, content: str = "msg") -> MemoryEntry:
    return MemoryEntry(timestamp=ts, role="user", content=content, source_tag="chat")


def _write_episodes(persona_dir: Path, entries: list[MemoryEntry]) -> None:
    persona_dir.mkdir(parents=True, exist_ok=True)
    text = "".join(serialize_entry(e) + "\n" for e in entries)
    (persona_dir / "episodes.jsonl").write_text(text, encoding="utf-8")


def _memory_layout(tmp_path: Path) -> tuple[Path, Path]:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    state_file = memory_dir / ".reflect-state.json"
    return memory_dir, state_file


def test_collect_returns_only_episodes_after_last_ts(tmp_path: Path) -> None:
    memory_dir, state_file = _memory_layout(tmp_path)
    persona_a = memory_dir / "persona-a"
    _write_episodes(persona_a, [_episode(T1), _episode(T2), _episode(T3)])
    save_reflect_state(state_file, {"persona-a": {"last_ts": T2, "failures": 0}})

    payload = build_collect_payload(memory_dir, state_file=state_file, now=NOW)

    assert payload["version"] == 1
    assert len(payload["items"]) == 1
    row = payload["items"][0]
    assert row["persona"] == "persona-a"
    assert [e["timestamp"] for e in row["episodes"]] == [T3]
    assert isinstance(row["active"], list)


def test_collect_skips_persona_without_new_episodes(tmp_path: Path) -> None:
    memory_dir, state_file = _memory_layout(tmp_path)
    _write_episodes(memory_dir / "persona-a", [_episode(T1)])
    save_reflect_state(state_file, {"persona-a": {"last_ts": T3, "failures": 0}})
    (memory_dir / "persona-b").mkdir()

    payload = build_collect_payload(memory_dir, state_file=state_file, now=NOW)

    assert payload["items"] == []


def test_collect_persona_filter(tmp_path: Path) -> None:
    memory_dir, state_file = _memory_layout(tmp_path)
    _write_episodes(memory_dir / "persona-a", [_episode(T3)])
    _write_episodes(memory_dir / "persona-b", [_episode(T3, "b")])

    payload = build_collect_payload(
        memory_dir,
        state_file=state_file,
        now=NOW,
        persona_filter=["persona-b"],
    )

    personas = [i["persona"] for i in payload["items"]]
    assert personas == ["persona-b"]


def test_collect_archives_episodes_older_than_90_days(tmp_path: Path) -> None:
    memory_dir, state_file = _memory_layout(tmp_path)
    old_ts = (NOW - timedelta(days=100)).strftime("%Y-%m-%d %H:%M")
    recent_ts = (NOW - timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
    persona_dir = memory_dir / "persona-a"
    _write_episodes(persona_dir, [_episode(old_ts, "old"), _episode(recent_ts, "recent")])

    payload = build_collect_payload(memory_dir, state_file=state_file, now=NOW)

    remaining = (persona_dir / "episodes.jsonl").read_text(encoding="utf-8")
    assert "old" not in remaining
    assert "recent" in remaining
    month = old_ts[:7]
    archive_path = persona_dir / "episodes" / f"{month}.jsonl"
    assert archive_path.is_file()
    archived = archive_path.read_text(encoding="utf-8")
    assert "old" in archived
    timestamps = [e["timestamp"] for i in payload["items"] for e in i["episodes"]]
    assert old_ts not in timestamps


def test_collect_lists_only_persona_directories(tmp_path: Path) -> None:
    memory_dir, state_file = _memory_layout(tmp_path)
    _write_episodes(memory_dir / "persona-a", [_episode(T3)])

    flat = memory_dir / "persona-c.jsonl"
    flat.write_text(serialize_entry(_episode(T3, "flat")) + "\n", encoding="utf-8")

    hidden = memory_dir / ".hidden"
    hidden.mkdir()
    _write_episodes(hidden, [_episode(T3, "hidden")])

    archive = memory_dir / "archive"
    archive.mkdir()
    _write_episodes(archive, [_episode(T3, "archive")])

    payload = build_collect_payload(memory_dir, state_file=state_file, now=NOW)

    personas = [i["persona"] for i in payload["items"]]
    assert personas == ["persona-a"]


def _valid_collect(persona: str = "persona-a") -> dict:
    return {
        "version": 1,
        "items": [
            {
                "persona": persona,
                "episodes": [serialize_entry(_episode(T3))],
                "active": [],
            }
        ],
    }


def _valid_llm(persona: str = "persona-a") -> dict:
    return {
        "version": 1,
        "items": [
            {
                "persona": persona,
                "reflection": "noted",
                "add": [{"kind": "fact", "content": "x", "subject": "user"}],
                "supersede": [],
            }
        ],
    }


def test_validate_llm_payload_rejects_collect_version_mismatch() -> None:
    collect = {"version": 0, "items": [{"persona": "persona-a", "episodes": [], "active": []}]}
    llm = {"version": 1, "items": [{"persona": "persona-a"}]}
    with pytest.raises(ValueError, match="collect version mismatch"):
        validate_llm_payload(collect, llm)


@pytest.mark.parametrize(
    "llm,match",
    [
        ({"version": 2, "items": []}, "llm version mismatch"),
        ({"version": 1, "items": "x"}, "llm.items must be a list"),
        ({"version": 1, "items": ["x"]}, "llm item must be object"),
        ({"version": 1, "items": [{"persona": ""}]}, "persona required"),
        ({"version": 1, "items": [{"persona": "persona-a"}, {"persona": "persona-a"}]}, "duplicate"),
        (
            {"version": 1, "items": [{"persona": "persona-b"}]},
            "persona set mismatch",
        ),
        (
            {
                "version": 1,
                "items": [
                    {"persona": "persona-a"},
                    {"persona": "persona-b"},
                ],
            },
            "persona set mismatch",
        ),
    ],
)
def test_validate_llm_payload_rejects_invalid(llm: dict, match: str) -> None:
    collect = _valid_collect()
    with pytest.raises(ValueError, match=match):
        validate_llm_payload(collect, llm)


def test_validate_llm_payload_accepts_valid() -> None:
    collect = _valid_collect()
    llm = _valid_llm()
    by_persona = validate_llm_payload(collect, llm)
    assert by_persona == {"persona-a": llm["items"][0]}


def test_apply_writes_semantic_and_advances_last_ts(tmp_path: Path) -> None:
    memory_dir, state_file = _memory_layout(tmp_path)
    persona_dir = memory_dir / "persona-a"
    persona_dir.mkdir()
    store = SemanticStore(persona_dir / "semantic.jsonl", persona_stem="persona-a")
    active = store.append("fact", "keep me", "user", "seed", ts=T1)
    collect = {
        "version": 1,
        "items": [
            {
                "persona": "persona-a",
                "episodes": [json.loads(serialize_entry(_episode(T3)))],
                "active": [],
            }
        ],
    }
    llm = {
        "version": 1,
        "items": [
            {
                "persona": "persona-a",
                "reflection": "learned",
                "add": [{"kind": "fact", "content": "new fact", "subject": "user"}],
                "supersede": [active.id],
            }
        ],
    }

    ok, err = apply_from_payloads(
        collect,
        llm,
        memory_dir=memory_dir,
        state_file=state_file,
        now=NOW,
    )

    assert ok is True
    assert err is None
    state = load_reflect_state(state_file)
    assert state["persona-a"]["last_ts"] == T3
    assert state["persona-a"]["failures"] == 0
    entries = store.active()
    kinds = {e.kind for e in entries}
    assert "reflection" in kinds
    assert any(e.content == "new fact" for e in entries)
    old = store.get(active.id)
    assert old is not None
    assert old.status == "superseded"


def test_apply_failure_notifies_once_at_threshold(tmp_path: Path) -> None:
    memory_dir, state_file = _memory_layout(tmp_path)
    persona_dir = memory_dir / "persona-a"
    _write_episodes(persona_dir, [_episode(T3)])
    collect = _valid_collect()
    bad_llm = {
        "version": 1,
        "items": [
            {
                "persona": "persona-a",
                "reflection": 1,
                "add": [],
                "supersede": [],
            }
        ],
    }
    notifier = MagicMock()

    for _ in range(4):
        ok, err = apply_from_payloads(
            collect,
            bad_llm,
            memory_dir=memory_dir,
            state_file=state_file,
            now=NOW,
            failure_threshold=3,
            on_failure=notifier,
        )
        assert ok is False
        assert err is None

    state = load_reflect_state(state_file)
    assert state["persona-a"]["failures"] == 4
    assert state["persona-a"]["last_ts"] is None
    notifier.assert_called_once()
    args = notifier.call_args[0]
    assert args[0] == "persona-a"
    assert args[1] == 3
    assert isinstance(args[2], str)


def test_apply_dry_run_writes_nothing(tmp_path: Path) -> None:
    memory_dir, state_file = _memory_layout(tmp_path)
    persona_dir = memory_dir / "persona-a"
    _write_episodes(persona_dir, [_episode(T3)])
    collect = _valid_collect()
    llm = _valid_llm()

    ok, err = apply_from_payloads(
        collect,
        llm,
        memory_dir=memory_dir,
        state_file=state_file,
        now=NOW,
        dry_run=True,
    )

    assert ok is True
    assert err is None
    assert not (persona_dir / "semantic.jsonl").exists()
    assert not state_file.exists()


def test_apply_persona_set_mismatch_returns_error(tmp_path: Path) -> None:
    memory_dir, state_file = _memory_layout(tmp_path)
    persona_dir = memory_dir / "persona-a"
    _write_episodes(persona_dir, [_episode(T3)])
    collect = _valid_collect()
    llm = {"version": 1, "items": [{"persona": "other", "reflection": "", "add": [], "supersede": []}]}

    ok, err = apply_from_payloads(
        collect,
        llm,
        memory_dir=memory_dir,
        state_file=state_file,
        now=NOW,
    )

    assert ok is False
    assert err is not None
    assert "persona set mismatch" in err
    assert not (persona_dir / "semantic.jsonl").exists()
    assert not state_file.exists()


def test_reflect_state_round_trip(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    assert load_reflect_state(missing) == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_reflect_state(broken) == {}

    state_file = tmp_path / "state.json"
    save_reflect_state(state_file, {"persona-a": {"last_ts": T2, "failures": 0}})

    loaded = json.loads(state_file.read_text(encoding="utf-8"))
    assert loaded == {"persona-a": {"last_ts": T2, "failures": 0}}
    assert not (tmp_path / "state.json.tmp").exists()
    assert load_reflect_state(state_file) == loaded


def test_collect_new_episodes_filters_by_last_ts(tmp_path: Path) -> None:
    persona_dir = tmp_path / "p"
    _write_episodes(persona_dir, [_episode(T1), _episode(T3)])
    assert collect_new_episodes(persona_dir, None) == [_episode(T1), _episode(T3)]
    assert collect_new_episodes(persona_dir, T2) == [_episode(T3)]
    assert collect_new_episodes(tmp_path / "missing", T2) == []
