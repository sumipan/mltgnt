"""Conversation-id-based thread_queue tests (#3317).

drain only returns TurnInput. It does not dispatch.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mltgnt.config import ConversationConfig
from mltgnt.config.language import EN, set_language_pack
from mltgnt.interfaces.turn import Attachment, TurnInput


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


def test_drained_author_is_set_only_for_a_single_author(queue_root: Path):
    """Issue #7323: drained_author and drain_to_turn_input author_id."""
    from mltgnt.conversation.thread_queue import admit, drain_to_turn_input, drained_author

    assert drained_author([]) is None
    assert drained_author([{"user": "alice"}, {"user": "alice"}]) == "alice"
    assert drained_author([{"user": "alice"}, {"user": "bob"}]) is None
    assert drained_author([{"user": "alice"}, {"user": ""}]) is None
    assert drained_author([{"text": "no user key"}]) is None

    cid = "C1:171.0000"
    assert admit(cid, "first", message_ts="171.0001", author="alice").proceed
    assert admit(cid, "second", message_ts="171.0002", author="alice").queued
    turn = drain_to_turn_input(cid)
    assert turn is not None
    assert turn.author_id == "alice"


def test_drain_to_turn_input_empty_returns_none(queue_root: Path):
    from mltgnt.conversation.thread_queue import admit, drain_to_turn_input

    assert admit("C1:171.0000", "only", message_ts="171.0001").proceed
    assert drain_to_turn_input("C1:171.0000") is None


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


def test_drain_to_turn_input_includes_attachments(queue_root: Path):
    from mltgnt.conversation.thread_queue import admit, drain_to_turn_input

    attachment = Attachment(name="a.png", content_type="image/png", uri="https://example/a")
    assert admit("C1:171.0000", "first", message_ts="171.0001").proceed
    assert admit(
        "C1:171.0000",
        "with file",
        message_ts="171.0002",
        attachments=(attachment,),
    ).queued
    turn = drain_to_turn_input("C1:171.0000")
    assert turn is not None
    assert turn.attachments == (attachment,)


def test_drained_attachments_tolerates_legacy_and_malformed_entries():
    from mltgnt.conversation.thread_queue import drained_attachments

    entries = [
        {"text": "legacy", "kind": "message"},
        {"text": "bad", "attachments": "not-a-list"},
        {"text": "mixed", "attachments": [{"name": ""}, "x", {"name": "ok.png", "uri": 1}]},
    ]
    assert drained_attachments(entries) == (
        Attachment(name="ok.png", content_type=None, uri=None),
    )
    assert drained_attachments([]) == ()


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
