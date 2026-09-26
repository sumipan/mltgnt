"""Public names and injection points of mltgnt.conversation modules (#4024).

Old private names stay as aliases of the same objects for one release.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mltgnt.config import ConversationConfig
from mltgnt.conversation import (
    session_compact,
    session_store,
    thread_index,
    thread_persona_store,
    thread_queue,
)


@pytest.fixture
def config(tmp_path: Path) -> ConversationConfig:
    return ConversationConfig(
        queue_dir=tmp_path / "queue",
        sessions_dir=tmp_path / "sessions",
        ledger_dir=tmp_path / "ledger",
        thread_index_dir=tmp_path / "thread-index",
        thread_persona_path=tmp_path / "thread-persona-map.jsonl",
        thread_persona_ttl_days=30,
    )


@pytest.mark.parametrize(
    "module,public,private",
    [
        (thread_queue, "thread_queue_config", "_thread_queue_config"),
        (thread_queue, "read_state", "_read_state"),
        (thread_queue, "is_stale", "_is_stale"),
        (thread_persona_store, "LOCK", "_LOCK"),
        (thread_persona_store, "ttl_days", "_ttl_days"),
        (thread_persona_store, "is_expired", "_is_expired"),
        (thread_persona_store, "compact", "_compact"),
        (session_compact, "estimate_tokens", "_estimate_tokens"),
        (session_compact, "format_turns_for_prompt", "_format_turns_for_prompt"),
        (session_compact, "build_prompt", "_build_prompt"),
        (session_compact, "audit_path", "_audit_path"),
        (session_store, "sessions_dir", "_sessions_dir"),
        (thread_index, "write_post_content", "_write_post_content"),
    ],
)
def test_private_name_is_alias_of_public(module, public: str, private: str) -> None:
    assert getattr(module, public) is getattr(module, private)
    assert public in module.__all__
    assert private not in module.__all__


@pytest.mark.parametrize(
    "module,name",
    [
        (thread_queue, "set_lock_registry"),
        (thread_queue, "set_config_provider"),
        (thread_index, "set_post_content_writer"),
    ],
)
def test_injection_points_are_public(module, name: str) -> None:
    assert callable(getattr(module, name))
    assert name in module.__all__


# --- thread_queue ---------------------------------------------------------


def test_set_config_provider_overrides_and_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(thread_queue, "_active_config", None)
    default = thread_queue.thread_queue_config()
    try:
        thread_queue.set_config_provider(
            lambda: {"stale_after_sec": 1, "max_queued": 2, "cleanup_ttl_days": 3}
        )
        assert thread_queue.thread_queue_config()["stale_after_sec"] == 1
    finally:
        thread_queue.set_config_provider(None)
    assert thread_queue.thread_queue_config() == default
    assert default["stale_after_sec"] == 3600


def test_config_provider_is_used_by_admit(
    config: ConversationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(thread_queue, "THREAD_QUEUE_DIR", config.queue_dir)
    monkeypatch.setattr(thread_queue, "_locks", {})
    monkeypatch.setattr(thread_queue, "_active_config", config)
    try:
        thread_queue.set_config_provider(
            lambda: {"stale_after_sec": 3600, "max_queued": 0, "cleanup_ttl_days": 14}
        )
        assert thread_queue.admit("C1:1.0", "a", message_ts="1.1").status == "accepted"
        assert thread_queue.admit("C1:1.0", "b", message_ts="1.2").status == "rejected"
    finally:
        thread_queue.set_config_provider(None)


def test_legacy_config_attribute_replacement_still_works(
    config: ConversationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(thread_queue, "THREAD_QUEUE_DIR", config.queue_dir)
    monkeypatch.setattr(thread_queue, "_locks", {})
    monkeypatch.setattr(
        thread_queue,
        "_thread_queue_config",
        lambda: {"stale_after_sec": 3600, "max_queued": 0, "cleanup_ttl_days": 14},
    )
    assert thread_queue.admit("C1:1.0", "a", message_ts="1.1").status == "accepted"
    assert thread_queue.admit("C1:1.0", "b", message_ts="1.2").status == "rejected"


def test_set_lock_registry_is_used(
    config: ConversationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(thread_queue, "THREAD_QUEUE_DIR", config.queue_dir)
    monkeypatch.setattr(thread_queue, "_locks", {})
    registry: dict[str, threading.Lock] = {}
    thread_queue.set_lock_registry(registry)
    thread_queue.record_job("C1:1.0", "uuid-1")
    assert list(registry) == [thread_queue.storage_key("C1:1.0")]
    assert thread_queue._locks is registry


def test_read_state_and_is_stale(
    config: ConversationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(thread_queue, "THREAD_QUEUE_DIR", config.queue_dir)
    assert thread_queue.read_state("missing")["status"] == "idle"
    old = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    assert thread_queue.is_stale({"status": "running", "started_at": old}, 5) is True
    assert thread_queue.is_stale({"status": "running", "started_at": old}, 60) is False
    assert thread_queue.is_stale({"status": "idle", "started_at": old}, 5) is False


# --- thread_persona_store -------------------------------------------------


def test_is_expired_boundary(config: ConversationConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(thread_persona_store, "_active_config", config)
    assert thread_persona_store.ttl_days() == 30
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    exact = ts + timedelta(days=30)
    assert thread_persona_store.is_expired(ts.isoformat(), now=exact) is False
    assert (
        thread_persona_store.is_expired(ts.isoformat(), now=exact + timedelta(seconds=1))
        is True
    )
    assert thread_persona_store.is_expired("not-a-ts", now=exact) is True


def test_compact_writes_entries(config: ConversationConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(thread_persona_store, "_active_config", config)
    monkeypatch.setattr(thread_persona_store, "_store_path_override", None)
    ts = datetime.now(timezone.utc).isoformat()
    thread_persona_store.compact({"C1": ("persona-a", ts)})
    assert thread_persona_store.load() == {"C1": "persona-a"}
    assert isinstance(thread_persona_store.LOCK, type(threading.Lock()))


# --- session_compact / session_store --------------------------------------


def test_session_compact_helpers(config: ConversationConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    turns = [
        {"role": "user", "content": "abc"},
        {"role": "assistant", "persona": "persona-a", "content": "def"},
    ]
    assert session_compact.estimate_tokens(turns) == 2.0
    formatted = session_compact.format_turns_for_prompt(turns, [{"content": "old"}])
    assert formatted.splitlines() == [
        "[compacted] old",
        "[user] abc",
        "[assistant:persona-a] def",
    ]
    assert session_compact.build_prompt(turns, []).endswith(
        session_compact.format_turns_for_prompt(turns, [])
    )
    monkeypatch.setattr(session_compact, "_active_config", config)
    assert session_compact.audit_path() == config.sessions_dir.parent / "audit.jsonl"


def test_sessions_dir(config: ConversationConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_store, "_active_config", config)
    monkeypatch.setattr(session_store, "_sessions_dir_override", None)
    assert session_store.sessions_dir() == config.sessions_dir


# --- thread_index ---------------------------------------------------------


def test_set_post_content_writer_overrides_and_resets(
    config: ConversationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(thread_index, "_active_config", config)
    calls: list[tuple[str, str]] = []

    def writer(uid: str, content: str) -> str:
        calls.append((uid, content))
        return f"custom/{uid}"

    try:
        thread_index.set_post_content_writer(writer)
        thread_index.register_bot_post("C1:1.0", "1.1", "uid-1", content="body")
        assert calls == [("uid-1", "body")]
        assert thread_index.lookup_result_path("C1:1.0", "1.1") == "custom/uid-1"
    finally:
        thread_index.set_post_content_writer(None)

    thread_index.register_bot_post("C1:1.0", "1.2", "uid-2", content="body2")
    assert len(calls) == 1
    default_path = config.thread_index_dir.parent / "posts" / "uid-2.md"
    assert default_path.read_text(encoding="utf-8") == "body2"
    assert thread_index.lookup_result_path("C1:1.0", "1.2") == str(default_path)


def test_legacy_post_writer_attribute_replacement_still_works(
    config: ConversationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(thread_index, "_active_config", config)
    monkeypatch.setattr(thread_index, "_write_post_content", lambda uid, content: "legacy")
    thread_index.register_bot_post("C1:1.0", "1.1", "uid-1", content="body")
    assert thread_index.lookup_result_path("C1:1.0", "1.1") == "legacy"
