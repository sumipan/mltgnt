"""tests/loops/test_store.py — 永続化テスト。"""
from __future__ import annotations

import json
from pathlib import Path

from mltgnt.loops.models import LoopState, state_from_json, state_to_json
from mltgnt.loops import store
import pytest


def _sample_state(loop_id: str = "test") -> LoopState:
    return LoopState(
        loop_id=loop_id,
        objective_path="/tmp/obj.md",
        objective_hash="abc",
        title="T",
        body="body",
        status="clarifying",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        created_at="2026-08-20T12:00:00+09:00",
        updated_at="2026-08-20T12:00:00+09:00",
    )


def test_atomic_state_roundtrip(tmp_path):
    state_dir = tmp_path / "state"
    state = _sample_state()
    store.save_state(state_dir, state)
    loaded = store.load_state(state_dir, "test")
    assert loaded is not None
    assert loaded.loop_id == "test"
    assert loaded.status == "clarifying"


def test_events_ignore_corrupt_trailing_line(tmp_path):
    state_dir = tmp_path / "state"
    events_path = store.loop_state_dir(state_dir, "test") / "events.jsonl"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        '{"ts":"t","loop_id":"test","iteration":1,"event":"a","data":{}}\n{broken',
        encoding="utf-8",
    )
    events = store.read_events(state_dir, "test")
    assert len(events) == 1


def test_inbox_consume(tmp_path):
    state_dir = tmp_path / "state"
    inbox = store._inbox_dir(state_dir, "test")
    inbox.mkdir(parents=True)
    (inbox / "001-msg.json").write_text(
        json.dumps(
            {
                "kind": "answer",
                "message_id": "m1",
                "question_id": "q1",
                "text": "yes",
                "received_at": "2026-08-20T12:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )
    msgs = store.list_inbox_messages(state_dir, "test")
    assert len(msgs) == 1
    store.consume_inbox_message(state_dir, "test", msgs[0].filename)
    assert not (inbox / "001-msg.json").exists()
    assert (store._consumed_dir(state_dir, "test") / "001-msg.json").exists()


def test_list_restorable_loops(tmp_path):
    state_dir = tmp_path / "state"
    store.save_state(state_dir, _sample_state("a"))
    assert "a" in store.list_restorable_loops(state_dir)


def test_archive_terminal_state_excluded_from_restorable(tmp_path):
    state_dir = tmp_path / "state"
    state = _sample_state("done-loop")
    state.status = "done"
    store.save_state(state_dir, state)

    dest = store.archive_terminal_state(state_dir, "done-loop")
    assert dest is not None
    assert dest.parent.name == "archive"
    assert dest.name.startswith("done-loop.archived-")
    assert not (state_dir / "done-loop").exists()
    assert "done-loop" not in store.list_restorable_loops(state_dir)
    assert "archive" not in store.list_restorable_loops(state_dir)


def test_state_load_rejects_wrong_required_type(tmp_path):
    state_dir = tmp_path / "state"
    state = _sample_state()
    data = state.to_dict()
    data["iteration"] = "1"
    path = store.loop_state_dir(state_dir, "test") / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="iteration must be int"):
        store.load_state(state_dir, "test")


def test_append_event_accepts_non_json_native_objects(tmp_path):
    """LLMResult / dataclass を含む data でも TypeError にならず JSONL に追記する。"""
    from dataclasses import dataclass

    from tests.loops.fakes import FakeLLMResult

    @dataclass
    class _Sample:
        name: str

    state_dir = tmp_path / "state"
    store.append_event(
        state_dir,
        "test",
        "llm_error",
        {
            "phase": "clarify",
            "result": FakeLLMResult(stdout="x", stderr="e", returncode=1),
            "payload": _Sample(name="n"),
        },
        iteration=1,
    )
    events = store.read_events(state_dir, "test")
    assert len(events) == 1
    assert events[0]["event"] == "llm_error"
    assert isinstance(events[0]["data"]["result"], str)
    assert isinstance(events[0]["data"]["payload"], str)


def test_initialize_deliverable_writes_body_with_trailing_newline(tmp_path):
    state_dir = tmp_path / "state"
    body = "alpha\nβ"
    path = store.initialize_deliverable(state_dir, "loop1", body)
    assert path == store.deliverable_path(state_dir, "loop1")
    assert path.read_text(encoding="utf-8") == "alpha\nβ\n"


def test_read_deliverable_excerpt_full_and_head_tail(tmp_path):
    state_dir = tmp_path / "state"
    store.initialize_deliverable(state_dir, "loop1", "abcdefghij")
    assert store.read_deliverable_excerpt(state_dir, "loop1", 100) == "abcdefghij\n"

    long_body = "A" * 50 + "B" * 50
    store.initialize_deliverable(state_dir, "loop1", long_body)
    excerpt = store.read_deliverable_excerpt(state_dir, "loop1", 40)
    assert len(excerpt) <= 40
    assert excerpt.startswith("A")
    assert "…" in excerpt
    assert excerpt.endswith("B") or excerpt.endswith("B\n")


def test_read_deliverable_excerpt_rejects_non_positive(tmp_path):
    state_dir = tmp_path / "state"
    store.initialize_deliverable(state_dir, "loop1", "x")
    with pytest.raises(ValueError, match="max_chars"):
        store.read_deliverable_excerpt(state_dir, "loop1", 0)


def test_deliverable_snapshot_utf8_sizes(tmp_path):
    state_dir = tmp_path / "state"
    store.initialize_deliverable(state_dir, "loop1", "α")
    snap = store.deliverable_snapshot(state_dir, "loop1")
    assert snap["path"] == str(store.deliverable_path(state_dir, "loop1"))
    # "α\n" → 2 chars; UTF-8 bytes = 2 (α) + 1 (\n) = 3
    assert snap["chars"] == 2
    assert snap["bytes"] == 3


def test_deliverable_uses_files_adapter(tmp_path, monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_write(path, content, *, repo_root=None):
        calls.append(("write", path))
        target = (repo_root or Path(".")) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def fake_read(path, *, repo_root=None):
        calls.append(("read", path))
        from types import SimpleNamespace

        text = ((repo_root or Path(".")) / path).read_text(encoding="utf-8")
        return SimpleNamespace(content=text)

    monkeypatch.setattr(store, "md_write", fake_write)
    monkeypatch.setattr(store, "md_read", fake_read)

    state_dir = tmp_path / "state"
    store.initialize_deliverable(state_dir, "loop1", "hello")
    store.read_deliverable_excerpt(state_dir, "loop1", 100)
    assert ("write", "deliverable.md") in calls
    assert ("read", "deliverable.md") in calls


def test_v0194_state_fixture_loads_with_defaults():
    """v0.19.4 形式（追加キーなし・schema_version: 1）をロードできる。"""
    data = {
        "loop_id": "legacy",
        "objective_path": "/tmp/obj.md",
        "objective_hash": "abc",
        "title": "T",
        "body": "body",
        "status": "executing",
        "iteration": 1,
        "max_iterations": 5,
        "persona": "mizuho",
        "schema_version": 1,
        "subtasks": [
            {
                "id": "s1",
                "title": "S1",
                "kind": "auto",
                "prompt": "do",
                "status": "pending",
                "result": "",
            }
        ],
    }
    state = LoopState.from_dict(data)
    assert state.schema_version == 1
    assert state.plan_approval is True
    assert state.plan_revision == 0
    assert state.replan_count == 0
    assert state.subtasks[0].depends == []  # legacy sequential: single task
    # two tasks without depends keys → sequential
    data2 = dict(data)
    data2["subtasks"] = [
        {"id": "s1", "title": "S1", "kind": "auto", "prompt": "a", "status": "success", "result": "ok"},
        {"id": "s2", "title": "S2", "kind": "human", "prompt": "b", "status": "pending", "result": ""},
    ]
    state2 = LoopState.from_dict(data2)
    assert state2.subtasks[0].depends == []
    assert state2.subtasks[1].depends == ["s1"]


def test_list_inbox_messages_includes_comment(tmp_path):
    state_dir = tmp_path / "state"
    inbox = store._inbox_dir(state_dir, "test")
    inbox.mkdir(parents=True)
    (inbox / "001-c.json").write_text(
        json.dumps(
            {
                "kind": "comment",
                "message_id": "c1",
                "question_id": "",
                "text": "hi",
                "received_at": "2026-08-20T12:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )
    (inbox / "002-a.json").write_text(
        json.dumps(
            {
                "kind": "answer",
                "message_id": "a1",
                "question_id": "q1",
                "text": "yes",
                "received_at": "2026-08-20T12:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )
    msgs = store.list_inbox_messages(state_dir, "test")
    kinds = {m.kind for m in msgs}
    assert kinds == {"comment", "answer"}


def test_list_inbox_messages_skips_corrupt_and_non_string(tmp_path):
    state_dir = tmp_path / "state"
    inbox = store._inbox_dir(state_dir, "test")
    inbox.mkdir(parents=True)
    (inbox / "001.json").write_text("{nope", encoding="utf-8")
    (inbox / "002.json").write_text(
        json.dumps(
            {
                "kind": "comment",
                "message_id": 1,
                "question_id": "",
                "text": "x",
                "received_at": "t",
            }
        ),
        encoding="utf-8",
    )
    (inbox / "003.json").write_text(
        json.dumps(
            {
                "kind": "comment",
                "message_id": "ok",
                "question_id": "",
                "text": "y",
                "received_at": "2026-08-20T12:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )
    msgs = store.list_inbox_messages(state_dir, "test")
    assert len(msgs) == 1
    assert msgs[0].message_id == "ok"
