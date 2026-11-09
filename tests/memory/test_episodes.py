"""Tests for mltgnt.memory.episodes (Issue #6238)."""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from mltgnt.config import MemoryConfig
from mltgnt.memory import append_turn_episodes, parse_jsonl, persona_memory_lock
from mltgnt.persona.memory import direct_dedupe_base


def _config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(chat_dir=tmp_path, chat_memory_dir=tmp_path / "memory", commit_debounce_sec=3600)


def test_append_turn_episodes_writes_direct_pair(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "memory" / "p").mkdir(parents=True)
    ok = append_turn_episodes(
        config,
        "p",
        user_text="hello",
        assistant_text="hi",
        space_id="C",
        thread_key="T",
        message_id="M1",
        timestamp="2026-09-27 10:00",
    )
    assert ok
    entries = parse_jsonl(tmp_path / "memory" / "p" / "episodes.jsonl")
    base = direct_dedupe_base("C", "T", "M1")
    keys = [e.dedupe_key for e in entries]
    assert keys == [f"{base}:user", f"{base}:assistant"]


def test_append_turn_episodes_is_idempotent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "memory" / "p").mkdir(parents=True)
    kwargs = dict(
        config=config,
        persona_stem="p",
        user_text="hello",
        assistant_text="hi",
        space_id="C",
        thread_key="T",
        message_id="M1",
        timestamp="2026-09-27 10:00",
    )
    assert append_turn_episodes(**kwargs)
    assert append_turn_episodes(**kwargs)
    assert len(parse_jsonl(tmp_path / "memory" / "p" / "episodes.jsonl")) == 2


def test_append_turn_episodes_prefixes_tool_trace(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "memory" / "p").mkdir(parents=True)
    append_turn_episodes(
        config,
        "p",
        user_text="run",
        assistant_text="done",
        space_id="C",
        thread_key="T",
        message_id="M2",
        timestamp="2026-09-27 10:00",
        tool_trace=[{"tool": "search", "args": {"q": "x"}, "result": "ok"}],
    )
    assistant = [e for e in parse_jsonl(tmp_path / "memory" / "p" / "episodes.jsonl") if e.role == "assistant"][0]
    assert assistant.content.startswith("[tool: search")
    assert assistant.content.endswith("done")
    assert assistant.content.count("[tool: search") == 1


def test_append_turn_episodes_returns_false_on_lock_timeout(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = MemoryConfig(chat_dir=tmp_path, chat_memory_dir=tmp_path / "memory", lock_timeout_sec=0.01)
    (tmp_path / "memory" / "p").mkdir(parents=True)
    with persona_memory_lock(config, "p"):
        with caplog.at_level(logging.WARNING, logger="mltgnt.memory.episodes"):
            ok = append_turn_episodes(
                config,
                "p",
                user_text="a",
                assistant_text="b",
                space_id="C",
                thread_key="T",
                message_id="M3",
                timestamp="2026-09-27 10:00",
                lock_timeout_sec=0.05,
            )
        assert not ok
    assert not (tmp_path / "memory" / "p" / "episodes.jsonl").exists()


def test_append_turn_episodes_retry_completes_partial_pair(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "memory" / "p").mkdir(parents=True)
    calls = {"n": 0}

    from mltgnt.memory.api import append_memory_entry as real_append

    def flaky(config, persona_stem, role, *args, **kwargs):
        calls["n"] += 1
        if role == "assistant" and calls["n"] == 2:
            return False
        return real_append(config, persona_stem, role, *args, **kwargs)

    with patch("mltgnt.memory.episodes.append_memory_entry", side_effect=flaky):
        assert not append_turn_episodes(
            config,
            "p",
            user_text="u",
            assistant_text="a",
            space_id="C",
            thread_key="T",
            message_id="M4",
            timestamp="2026-09-27 10:00",
        )
    assert append_turn_episodes(
        config,
        "p",
        user_text="u",
        assistant_text="a",
        space_id="C",
        thread_key="T",
        message_id="M4",
        timestamp="2026-09-27 10:00",
    )
    entries = parse_jsonl(tmp_path / "memory" / "p" / "episodes.jsonl")
    base = direct_dedupe_base("C", "T", "M4")
    assert len(entries) == 2
    assert {e.dedupe_key for e in entries} == {f"{base}:user", f"{base}:assistant"}
