"""Conversation-id-based thread_queue tests (#3317).

drain only returns TurnInput. It does not dispatch.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mltgnt.config import ConversationConfig
from mltgnt.config.language import EN, set_language_pack
from mltgnt.interfaces.turn import TurnInput


@pytest.fixture
def queue_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "conversation-threads"
    root.mkdir(parents=True)
    config = ConversationConfig(
        queue_dir=root,
        sessions_dir=tmp_path / "sessions",
        ledger_dir=tmp_path / "ledger",
        thread_index_dir=tmp_path / "thread-index",
        thread_persona_path=tmp_path / "thread-persona-map.jsonl",
    )
    import mltgnt.conversation.thread_queue as tq

    monkeypatch.setattr(tq, "THREAD_QUEUE_DIR", root)
    monkeypatch.setattr(tq, "_locks", {})
    monkeypatch.setattr(tq, "_active_config", config)
    monkeypatch.setattr(
        tq,
        "thread_queue_config",
        lambda: {"stale_after_sec": 3600, "max_queued": 20, "cleanup_ttl_days": 14},
    )
    return root


def test_admit_idle_to_running(queue_root: Path):
    from mltgnt.conversation.thread_queue import admit

    result = admit("C1:171.0000", "hello", message_ts="171.0001")
    assert result.proceed is True
    assert result.queued is False
    assert result.status == "accepted"
    state = json.loads((queue_root / "C1-171.0000" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "running"


def test_admit_running_queues(queue_root: Path):
    from mltgnt.conversation.thread_queue import admit

    thread_dir = queue_root / "C1-171.0000"
    thread_dir.mkdir(parents=True)
    (thread_dir / "state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "current_uuid": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = admit("C1:171.0000", "wait", message_ts="171.0002", author="U1")
    assert result.proceed is False
    assert result.queued is True
    assert result.status == "queued"
    inbox = list((thread_dir / "inbox").glob("*.json"))
    assert len(inbox) == 1
    payload = json.loads(inbox[0].read_text(encoding="utf-8"))
    assert payload["conversation_id"] == "C1:171.0000"
    assert "channel" not in payload


def test_drain_to_turn_input_returns_turn_input(queue_root: Path):
    from mltgnt.conversation.thread_queue import admit, drain_to_turn_input

    assert admit("C1:171.0000", "first", message_ts="171.0001").proceed
    assert admit("C1:171.0000", "second", message_ts="171.0002").queued
    assert admit("C1:171.0000", "third", message_ts="171.0003").queued

    turn = drain_to_turn_input("C1:171.0000", persona_id="p1")
    assert isinstance(turn, TurnInput)
    assert turn.conversation_id == "C1:171.0000"
    assert turn.persona_id == "p1"
    assert "second" in turn.text
    assert "third" in turn.text
    assert "[1] second" in turn.text


def test_drain_to_turn_input_empty_returns_none(queue_root: Path):
    from mltgnt.conversation.thread_queue import admit, drain_to_turn_input

    assert admit("C1:171.0000", "only", message_ts="171.0001").proceed
    assert drain_to_turn_input("C1:171.0000") is None


def test_module_has_no_ghdag_jobs_or_outbound():
    text = Path("src/mltgnt/conversation/thread_queue.py").read_text(encoding="utf-8")
    assert "ghdag" not in text
    assert "jobs/" not in text
    assert "slack_sdk" not in text
    assert "outbound" not in text
    assert "_run_triage_and_dispatch" not in text
    assert "load_secretary_yaml" not in text


def test_paths_come_from_conversation_config(tmp_path: Path):
    from mltgnt.conversation import configure, thread_queue as tq

    root = tmp_path / "q"
    configure(
        ConversationConfig(
            queue_dir=root,
            sessions_dir=tmp_path / "sessions",
            ledger_dir=tmp_path / "ledger",
            thread_index_dir=tmp_path / "thread-index",
            thread_persona_path=tmp_path / "thread-persona-map.jsonl",
        )
    )
    tq._locks.clear()
    result = tq.admit("C9:1.0", "hi", message_ts="1.1")
    assert result.status == "accepted"
    assert (root / "C9-1.0" / "state.json").is_file()


@pytest.fixture
def restore_language_pack():
    yield
    set_language_pack(EN)


def _cancel_pack():
    return replace(
        EN,
        cancel_words=frozenset({"abort"}),
        composite_header="CUSTOM HEADER: messages arrived while busy.",
        composite_cancel_suffix="CUSTOM SUFFIX: a cancel was requested.",
    )


def test_default_pack_cancel_words_are_en():
    from mltgnt.conversation.thread_queue import _is_cancel_instruction

    assert "cancel" in EN.cancel_words
    for word in EN.cancel_words:
        assert _is_cancel_instruction(word) is True
    assert _is_cancel_instruction("abort") is False


def test_default_pack_admit_marks_cancel_entry(queue_root: Path):
    from mltgnt.conversation.thread_queue import admit

    assert admit("C1:171.0000", "first", message_ts="171.0001").proceed
    assert admit("C1:171.0000", "cancel", message_ts="171.0002").queued
    inbox = list((queue_root / "C1-171.0000" / "inbox").glob("*.json"))
    assert len(inbox) == 1
    payload = json.loads(inbox[0].read_text(encoding="utf-8"))
    assert payload["kind"] == "cancel"


def test_default_pack_composite_instruction_uses_en():
    from mltgnt.conversation.thread_queue import build_composite_instruction

    plain = build_composite_instruction([{"text": "hello", "kind": "message"}])
    assert plain.startswith(EN.composite_header)
    assert "[1] hello" in plain
    assert EN.composite_cancel_suffix not in plain

    with_cancel = build_composite_instruction(
        [{"text": "hello", "kind": "message"}, {"text": "cancel", "kind": "cancel"}]
    )
    assert with_cancel.startswith(EN.composite_header)
    assert with_cancel.endswith(EN.composite_cancel_suffix)


def test_replaced_pack_changes_cancel_words(restore_language_pack):
    from mltgnt.conversation.thread_queue import _is_cancel_instruction

    set_language_pack(_cancel_pack())
    assert _is_cancel_instruction("abort") is True
    assert _is_cancel_instruction("cancel") is False
    assert _is_cancel_instruction("stop") is False


def test_replaced_pack_admit_marks_cancel_entry(queue_root: Path, restore_language_pack):
    from mltgnt.conversation.thread_queue import admit

    set_language_pack(_cancel_pack())
    assert admit("C1:171.0000", "first", message_ts="171.0001").proceed
    assert admit("C1:171.0000", "abort", message_ts="171.0002").queued
    assert admit("C1:171.0000", "cancel", message_ts="171.0003").queued
    inbox = (queue_root / "C1-171.0000" / "inbox").glob("*.json")
    kinds = {
        payload["text"]: payload["kind"]
        for payload in (json.loads(p.read_text(encoding="utf-8")) for p in inbox)
    }
    assert kinds == {"abort": "cancel", "cancel": "message"}


def test_replaced_pack_changes_composite_instruction(restore_language_pack):
    from mltgnt.conversation.thread_queue import build_composite_instruction

    pack = _cancel_pack()
    set_language_pack(pack)
    out = build_composite_instruction(
        [{"text": "hello", "kind": "message"}, {"text": "abort", "kind": "cancel"}]
    )
    assert out.startswith(pack.composite_header)
    assert out.endswith(pack.composite_cancel_suffix)
    assert EN.composite_header not in out
    assert EN.composite_cancel_suffix not in out
