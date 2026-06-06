"""tests/memory/dream/test_synthesizer.py — Synthesizer / dream api のテスト。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mltgnt.memory._format import MemoryEntry
from mltgnt.memory.dream import (
    DreamSection,
    DreamSummary,
    Synthesizer,
    read_dream,
    write_dream,
)


def _entry(content: str = "メモリ内容") -> MemoryEntry:
    return MemoryEntry(
        timestamp="2026-06-01 10:00",
        role="user",
        content=content,
        source_tag="chat",
    )


def _llm_response() -> str:
    return (
        "## 行動パターン\n"
        "朝型で作業を始める。\n\n"
        "## 好み・傾向\n"
        "簡潔な説明を好む。"
    )


def test_synthesize_creates_dream_summary_with_mock_llm() -> None:
    summary = Synthesizer.synthesize(
        [_entry()],
        None,
        persona="alice",
        llm_call=lambda _prompt: _llm_response(),
    )
    assert summary.persona == "alice"
    assert len(summary.sections) == 2
    assert summary.sections[0].category == "行動パターン"
    assert summary.sections[0].source_entries == 1
    assert summary.updated_at.endswith("+09:00") or "T" in summary.updated_at


def test_synthesize_merges_existing_sections() -> None:
    existing = DreamSummary(
        persona="alice",
        sections=[
            DreamSection(category="行動パターン", content="旧パターン", source_entries=2),
            DreamSection(category="保留カテゴリ", content="残す", source_entries=1),
        ],
        updated_at="2026-01-01T00:00:00+09:00",
    )

    def llm(_prompt: str) -> str:
        return "## 行動パターン\n新パターン\n\n## 好み・傾向\n新しい好み"

    summary = Synthesizer.synthesize(
        [_entry("更新後")],
        existing,
        persona="alice",
        llm_call=llm,
    )
    by_cat = {s.category: s.content for s in summary.sections}
    assert by_cat["行動パターン"] == "新パターン"
    assert by_cat["好み・傾向"] == "新しい好み"
    assert by_cat["保留カテゴリ"] == "残す"


def test_synthesize_raises_on_llm_failure() -> None:
    def failing_llm(_prompt: str) -> str:
        raise RuntimeError("LLM unavailable")

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        Synthesizer.synthesize(
            [_entry()],
            None,
            persona="alice",
            llm_call=failing_llm,
        )


def test_read_write_dream_roundtrip(tmp_path: Path) -> None:
    persona_dir = tmp_path / "alice"
    original = DreamSummary(
        persona="alice",
        sections=[DreamSection(category="行動パターン", content="text", source_entries=3)],
        updated_at="2026-06-07T12:00:00+09:00",
    )
    write_dream(persona_dir, original)
    loaded = read_dream(persona_dir)
    assert loaded == original


def test_write_dream_uses_atomic_replace(tmp_path: Path) -> None:
    persona_dir = tmp_path / "alice"
    summary = DreamSummary(
        persona="alice",
        sections=[DreamSection(category="行動パターン", content="a", source_entries=1)],
        updated_at="2026-06-07T12:00:00+09:00",
    )
    write_dream(persona_dir, summary)
    dream_path = persona_dir / "memory" / "dream.json"
    assert dream_path.is_file()
    assert not dream_path.with_suffix(".json.tmp").exists()
    data = json.loads(dream_path.read_text(encoding="utf-8"))
    assert data["persona"] == "alice"


def test_read_dream_returns_none_for_invalid_json(tmp_path: Path) -> None:
    persona_dir = tmp_path / "alice"
    dream_path = persona_dir / "memory" / "dream.json"
    dream_path.parent.mkdir(parents=True)
    dream_path.write_text("{not json", encoding="utf-8")
    assert read_dream(persona_dir) is None


def test_synthesize_writes_via_api(tmp_path: Path) -> None:
    summary = Synthesizer.synthesize(
        [_entry()],
        None,
        persona="alice",
        llm_call=lambda _prompt: _llm_response(),
    )
    write_dream(tmp_path / "alice", summary)
    loaded = read_dream(tmp_path / "alice")
    assert loaded is not None
    assert loaded.persona == "alice"
