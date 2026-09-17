"""tests/memory/dream/test_global_synthesis.py — tests for global.json cross-persona synthesis."""
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
        DreamSection(category="Behavior patterns", content=f"{prefix} behavior", source_entries=1),
        DreamSection(category="Preferences", content=f"{prefix} preference", source_entries=1),
    ]


def _global_llm_response() -> str:
    return (
        "## Behavior patterns\n"
        "Merged behavior patterns.\n\n"
        "## Preferences\n"
        "Merged preferences."
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
    assert summary.sections[0].category == "Behavior patterns"
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
    # Japanese text intentionally kept for CJK processing test
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

    # Japanese text intentionally kept for CJK processing test
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
            DreamSection(category="Behavior patterns", content="old global behavior", source_entries=1),
            DreamSection(category="retained category", content="keep", source_entries=1),
        ],
        updated_at="2026-01-01T00:00:00+09:00",
    )
    write_global(chat_dir, existing)

    def llm(_prompt: str) -> str:
        return "## Behavior patterns\nnew global behavior\n\n## Preferences\nnew global preference"

    config = _memory_config(chat_dir)
    summary = Synthesizer.synthesize_global(config, llm_call=llm)

    by_cat = {s.category: s.content for s in summary.sections}
    assert by_cat["Behavior patterns"] == "new global behavior"
    assert by_cat["Preferences"] == "new global preference"
    assert by_cat["retained category"] == "keep"


def test_read_global_summary_formats_like_read_dream_summary(chat_dir: Path) -> None:
    _write_persona_md(chat_dir, "alice")
    summary = DreamSummary(
        persona="__global__",
        sections=[
            DreamSection(category="Behavior patterns", content="merged behavior", source_entries=2),
            DreamSection(category="Preferences", content="merged preference", source_entries=1),
        ],
        updated_at="2026-06-07T12:00:00+09:00",
    )
    write_global(chat_dir, summary)

    config = _memory_config(chat_dir)
    result = read_global_summary(config)

    # Japanese text intentionally kept for CJK processing test
    assert result.startswith("\n\n## 記憶の要約\n\n")
    assert "### Behavior patterns\nmerged behavior" in result
    assert "### Preferences\nmerged preference" in result


def test_read_global_summary_returns_empty_when_missing(chat_dir: Path) -> None:
    config = _memory_config(chat_dir)
    assert read_global_summary(config) == ""


def test_write_global_uses_atomic_replace(chat_dir: Path) -> None:
    summary = DreamSummary(
        persona="__global__",
        sections=[DreamSection(category="Behavior patterns", content="a", source_entries=1)],
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
