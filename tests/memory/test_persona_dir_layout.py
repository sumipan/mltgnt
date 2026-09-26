"""Tests for persona-directory resolution in memory_file_path (Issue #4037)."""
from __future__ import annotations

from pathlib import Path

from mltgnt.config import MemoryConfig
from mltgnt.config.language import JA, LanguagePack
from mltgnt.memory import append_memory_entry, memory_file_path, parse_jsonl, persona_memory_lock


def _config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(chat_dir=tmp_path, chat_memory_dir=tmp_path / "memory")


def test_directory_layout_wins(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "memory" / "p").mkdir(parents=True)
    (tmp_path / "memory" / "p.jsonl").write_text("", encoding="utf-8")
    assert memory_file_path(config, "p") == tmp_path / "memory" / "p" / "episodes.jsonl"


def test_flat_only_keeps_legacy_path(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "p.jsonl").write_text("", encoding="utf-8")
    assert memory_file_path(config, "p") == tmp_path / "memory" / "p.jsonl"


def test_neither_keeps_legacy_path(tmp_path: Path) -> None:
    # Direct writers of memory_file_path() rely on the parent already existing,
    # so a persona without any file stays on the flat layout until migrated.
    config = _config(tmp_path)
    assert memory_file_path(config, "p") == tmp_path / "memory" / "p.jsonl"


def test_append_into_directory_layout(tmp_path: Path) -> None:
    config = MemoryConfig(chat_dir=tmp_path, chat_memory_dir=tmp_path / "memory", commit_debounce_sec=3600)
    (tmp_path / "memory" / "p").mkdir(parents=True)
    ok = append_memory_entry(config, "p", "user", "hello", "2026-09-26 10:00", source_tag="chat")
    assert ok
    path = tmp_path / "memory" / "p" / "episodes.jsonl"
    assert [e.content for e in parse_jsonl(path)] == ["hello"]
    assert not (tmp_path / "memory" / "p.jsonl").exists()


def test_lock_path_unchanged(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "memory" / "p").mkdir(parents=True)
    with persona_memory_lock(config, "p") as ok:
        assert ok
        assert (tmp_path / ".lock-memory-p").exists()


def test_language_pack_trigger_words_default_empty() -> None:
    assert JA.remember_trigger_words == frozenset()
    assert JA.forget_trigger_words == frozenset()
    fields = LanguagePack.__dataclass_fields__
    assert "remember_trigger_words" in fields and "forget_trigger_words" in fields
