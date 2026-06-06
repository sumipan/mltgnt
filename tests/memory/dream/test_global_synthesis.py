"""tests/memory/dream/test_global_synthesis.py — global.json 横断合成のテスト。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mltgnt.config import MemoryConfig
from mltgnt.memory.dream import DreamSection, DreamSummary, Synthesizer
from mltgnt.memory.dream.api import (
    global_json_path,
    read_global,
    read_global_summary,
    read_dream,
    write_dream,
    write_global,
)


def _write_persona_md(chat_dir: Path, stem: str) -> None:
    (chat_dir / f"{stem}.md").write_text(f"# {stem}\n", encoding="utf-8")


def _write_persona_dream(
    chat_dir: Path,
    stem: str,
    sections: list[DreamSection],
    *,
    memory_dir_name: str = "memory",
) -> None:
    summary = DreamSummary(
        persona=stem,
        sections=sections,
        updated_at="2026-06-07T12:00:00+09:00",
    )
    write_dream(chat_dir / stem, summary, memory_dir_name=memory_dir_name)


def _two_category_sections(prefix: str) -> list[DreamSection]:
    return [
        DreamSection(category="行動パターン", content=f"{prefix}の行動", source_entries=1),
        DreamSection(category="好み・傾向", content=f"{prefix}の好み", source_entries=1),
    ]


def _global_llm_response() -> str:
    return (
        "## 行動パターン\n"
        "統合された行動パターン。\n\n"
        "## 好み・傾向\n"
        "統合された好み・傾向。"
    )


@pytest.fixture
def chat_dir(tmp_path: Path) -> Path:
    agents = tmp_path / "agents"
    agents.mkdir()
    return agents


def _memory_config(chat_dir: Path, **kwargs: object) -> MemoryConfig:
    return MemoryConfig(chat_dir=chat_dir, **kwargs)  # type: ignore[arg-type]


def test_synthesize_global_three_personas(chat_dir: Path) -> None:
    for stem in ("alice", "bob", "charlie"):
        _write_persona_md(chat_dir, stem)
        _write_persona_dream(chat_dir, stem, _two_category_sections(stem))

    config = _memory_config(chat_dir)
    summary = Synthesizer.synthesize_global(
        config,
        llm_call=lambda _prompt: _global_llm_response(),
    )

    assert summary.persona == "__global__"
    assert len(summary.sections) == 2
    assert summary.sections[0].category == "行動パターン"
    assert summary.updated_at.endswith("+09:00") or "T" in summary.updated_at

    write_global(chat_dir, summary)
    loaded = read_global(chat_dir)
    assert loaded is not None
    assert loaded.persona == "__global__"
    assert global_json_path(chat_dir).is_file()


def test_synthesize_global_single_persona(chat_dir: Path) -> None:
    _write_persona_md(chat_dir, "alice")
    _write_persona_dream(chat_dir, "alice", _two_category_sections("alice"))

    config = _memory_config(chat_dir)
    summary = Synthesizer.synthesize_global(
        config,
        llm_call=lambda _prompt: _global_llm_response(),
    )

    assert summary.persona == "__global__"
    assert len(summary.sections) == 2


def test_synthesize_global_skips_personas_without_dream(chat_dir: Path) -> None:
    for stem in ("alice", "bob", "charlie"):
        _write_persona_md(chat_dir, stem)
    _write_persona_dream(chat_dir, "alice", _two_category_sections("alice"))
    _write_persona_dream(chat_dir, "bob", _two_category_sections("bob"))

    captured_prompts: list[str] = []

    def llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return _global_llm_response()

    config = _memory_config(chat_dir)
    summary = Synthesizer.synthesize_global(config, llm_call=llm)

    assert summary.persona == "__global__"
    assert "【ペルソナ: alice】" in captured_prompts[0]
    assert "【ペルソナ: bob】" in captured_prompts[0]
    assert "【ペルソナ: charlie】" not in captured_prompts[0]
    assert read_dream(chat_dir / "charlie") is None


def test_synthesize_global_respects_exclude_personas(chat_dir: Path) -> None:
    for stem in ("alice", "bob"):
        _write_persona_md(chat_dir, stem)
        _write_persona_dream(chat_dir, stem, _two_category_sections(stem))

    captured_prompts: list[str] = []

    def llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return _global_llm_response()

    config = _memory_config(chat_dir, global_dream_exclude_personas=("alice",))
    Synthesizer.synthesize_global(config, llm_call=llm)

    assert "【ペルソナ: alice】" not in captured_prompts[0]
    assert "【ペルソナ: bob】" in captured_prompts[0]


def test_synthesize_global_raises_when_no_dreams(chat_dir: Path) -> None:
    _write_persona_md(chat_dir, "alice")

    config = _memory_config(chat_dir)
    with pytest.raises(
        ValueError,
        match="no persona dream summaries available for global synthesis",
    ):
        Synthesizer.synthesize_global(
            config,
            llm_call=lambda _prompt: _global_llm_response(),
        )


def test_synthesize_global_raises_when_all_excluded(chat_dir: Path) -> None:
    _write_persona_md(chat_dir, "alice")
    _write_persona_dream(chat_dir, "alice", _two_category_sections("alice"))

    config = _memory_config(chat_dir, global_dream_exclude_personas=("alice",))
    with pytest.raises(
        ValueError,
        match="no persona dream summaries available for global synthesis",
    ):
        Synthesizer.synthesize_global(
            config,
            llm_call=lambda _prompt: _global_llm_response(),
        )


def test_synthesize_global_merges_existing_global(chat_dir: Path) -> None:
    _write_persona_md(chat_dir, "alice")
    _write_persona_dream(chat_dir, "alice", _two_category_sections("alice"))

    existing = DreamSummary(
        persona="__global__",
        sections=[
            DreamSection(category="行動パターン", content="旧global行動", source_entries=1),
            DreamSection(category="保留カテゴリ", content="残す", source_entries=1),
        ],
        updated_at="2026-01-01T00:00:00+09:00",
    )
    write_global(chat_dir, existing)

    def llm(_prompt: str) -> str:
        return "## 行動パターン\n新global行動\n\n## 好み・傾向\n新global好み"

    config = _memory_config(chat_dir)
    summary = Synthesizer.synthesize_global(config, llm_call=llm)

    by_cat = {s.category: s.content for s in summary.sections}
    assert by_cat["行動パターン"] == "新global行動"
    assert by_cat["好み・傾向"] == "新global好み"
    assert by_cat["保留カテゴリ"] == "残す"


def test_read_global_summary_formats_like_read_dream_summary(chat_dir: Path) -> None:
    _write_persona_md(chat_dir, "alice")
    summary = DreamSummary(
        persona="__global__",
        sections=[
            DreamSection(category="行動パターン", content="統合行動", source_entries=2),
            DreamSection(category="好み・傾向", content="統合好み", source_entries=1),
        ],
        updated_at="2026-06-07T12:00:00+09:00",
    )
    write_global(chat_dir, summary)

    config = _memory_config(chat_dir)
    result = read_global_summary(config)

    assert result.startswith("\n\n## 記憶の要約\n\n")
    assert "### 行動パターン\n統合行動" in result
    assert "### 好み・傾向\n統合好み" in result


def test_read_global_summary_returns_empty_when_missing(chat_dir: Path) -> None:
    config = _memory_config(chat_dir)
    assert read_global_summary(config) == ""


def test_write_global_uses_atomic_replace(chat_dir: Path) -> None:
    summary = DreamSummary(
        persona="__global__",
        sections=[DreamSection(category="行動パターン", content="a", source_entries=1)],
        updated_at="2026-06-07T12:00:00+09:00",
    )
    write_global(chat_dir, summary)
    path = global_json_path(chat_dir)
    assert path.is_file()
    assert not path.with_suffix(".json.tmp").exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["persona"] == "__global__"


def test_read_global_returns_none_for_invalid_json(chat_dir: Path) -> None:
    path = global_json_path(chat_dir)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert read_global(chat_dir) is None
