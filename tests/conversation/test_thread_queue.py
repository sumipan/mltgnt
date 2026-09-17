"""Conversation-id-based thread_queue tests (#3317).

drain only returns TurnInput. It does not dispatch.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mltgnt.config import ConversationConfig
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
        "_thread_queue_config",
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
    assert "処理中に以下の発言がありました" in turn.text  # Japanese text intentionally kept for CJK processing test


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
