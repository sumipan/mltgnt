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
