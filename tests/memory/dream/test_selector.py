"""tests/memory/dream/test_selector.py — DreamSelector のテスト。"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mltgnt.memory.dream import DreamSelector, DreamSection, DreamSummary, write_dream


def _write_jsonl(path: Path, content: str = '{"timestamp":"2026-06-01 10:00","role":"user","content":"hi","source_tag":"chat"}\n') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_dream(persona_dir: Path, updated_at: str) -> None:
    summary = DreamSummary(
        persona=persona_dir.name,
        sections=[DreamSection(category="行動パターン", content="old", source_entries=1)],
        updated_at=updated_at,
    )
    write_dream(persona_dir, summary)


def test_pick_targets_includes_persona_when_dream_missing(tmp_path: Path) -> None:
    persona_dir = tmp_path / "alice"
    _write_jsonl(persona_dir / "memory" / "chat.jsonl")
    result = DreamSelector.pick_targets([persona_dir])
    assert result == [persona_dir]


def test_pick_targets_includes_persona_when_jsonl_newer_than_dream(tmp_path: Path) -> None:
    persona_dir = tmp_path / "bob"
    memory_dir = persona_dir / "memory"
    _write_dream(persona_dir, "2020-01-01T00:00:00+09:00")
    jsonl_path = memory_dir / "chat.jsonl"
    _write_jsonl(jsonl_path)
    past = datetime(2019, 1, 1, tzinfo=ZoneInfo("Asia/Tokyo")).timestamp()
    os.utime(jsonl_path, (past + 100, past + 100))
    time.sleep(0.01)
    os.utime(jsonl_path, None)
    result = DreamSelector.pick_targets([persona_dir])
    assert result == [persona_dir]


def test_pick_targets_excludes_persona_when_dream_is_up_to_date(tmp_path: Path) -> None:
    persona_dir = tmp_path / "carol"
    memory_dir = persona_dir / "memory"
    jsonl_path = memory_dir / "chat.jsonl"
    _write_jsonl(jsonl_path)
    _write_dream(persona_dir, "2099-01-01T00:00:00+09:00")
    result = DreamSelector.pick_targets([persona_dir])
    assert result == []


def test_pick_targets_excludes_persona_without_jsonl(tmp_path: Path) -> None:
    persona_dir = tmp_path / "dave"
    (persona_dir / "memory").mkdir(parents=True)
    result = DreamSelector.pick_targets([persona_dir])
    assert result == []
