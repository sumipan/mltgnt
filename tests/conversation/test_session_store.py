"""Conversation-id-based session_store tests (#3317).

Do not import ghdag; resume support is decided via injected callbacks.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from mltgnt.config import ConversationConfig


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    ledger_dir = tmp_path / "ledger"
    sessions_dir.mkdir()
    ledger_dir.mkdir()
    config = ConversationConfig(
        queue_dir=tmp_path / "queue",
        sessions_dir=sessions_dir,
        ledger_dir=ledger_dir,
        thread_index_dir=tmp_path / "thread-index",
        thread_persona_path=tmp_path / "thread-persona-map.jsonl",
    )
    import mltgnt.conversation as conversation
    import mltgnt.conversation.session_store as ss

    conversation.configure(config)
    prev_check = ss._resume_check
    ss.configure_resume_check(None)
    yield {"sessions": sessions_dir, "ledger": ledger_dir}
    ss.configure_resume_check(prev_check)


def test_append_and_load_turns_by_conversation_id(isolated_store):
    from mltgnt.conversation import session_store as ss

    cid = "C123:1.0"
    ss.append_turn(cid, role="user", content="hello")
    ss.append_turn(cid, role="assistant", content="hi", persona_id="p1")
    turns = ss.load_turns(cid)
    assert len(turns) == 2
    assert turns[0]["role"] == "user"
    assert turns[1]["persona"] == "p1"
    assert ss.session_exists(cid)


def test_record_and_latest_session_local_record(isolated_store):
    from mltgnt.conversation import session_store as ss

    cid = "C123:1.0"
    ss.record_session(cid, "claude", "sess-abc")
    record = ss.latest_session(cid)
    assert record is not None
    assert record.engine == "claude"
    assert record.session_id == "sess-abc"
    assert ss.invalidate_session(cid) is True
    assert ss.latest_session(cid) is None


def test_resume_supported_uses_injected_callback(isolated_store):
    from mltgnt.conversation import session_store as ss

    calls: list[str] = []

    def _check(engine: str) -> bool:
        calls.append(engine)
        return engine == "cursor"

    ss.configure_resume_check(_check)
    assert ss.resume_supported("cursor") is True
    assert ss.resume_supported("claude") is False
    assert ss.resume_supported("") is False
    assert calls == ["cursor", "claude"]


def test_resume_supported_without_callback_is_false(isolated_store):
    from mltgnt.conversation import session_store as ss

    assert ss.resume_supported("claude") is False


def test_module_has_no_ghdag_or_jobs_literal():
    text = Path("src/mltgnt/conversation/session_store.py").read_text(encoding="utf-8")
    assert "ghdag" not in text
    assert "jobs/" not in text
    assert "slack_sdk" not in text


def test_gc_sessions(isolated_store):
    from mltgnt.conversation import session_store as ss

    ss.record_session("C1:1.0", "claude", "old")
    deleted = ss.gc_sessions(max_age=timedelta(seconds=0))
    assert deleted == 1


def test_types_reexport_interfaces_turn():
    from mltgnt.conversation import Attachment, HistoryMessage, TurnInput, TurnResult
    from mltgnt.interfaces import turn as turn_mod

    assert TurnInput is turn_mod.TurnInput
    assert TurnResult is turn_mod.TurnResult
    assert Attachment is turn_mod.Attachment
    assert HistoryMessage is turn_mod.HistoryMessage


def test_public_api_exports():
    import mltgnt.conversation as conversation

    for name in (
        "thread_queue",
        "thread_persona_store",
        "session_store",
        "thread_index",
        "session_compact",
        "fake_media",
        "configure",
        "ConversationConfig",
    ):
        assert hasattr(conversation, name), name
