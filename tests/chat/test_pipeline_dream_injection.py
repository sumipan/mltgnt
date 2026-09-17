"""Tests for read_dream_summary and dream injection via run_pipeline."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from mltgnt.config import MemoryConfig
from mltgnt.memory.dream import DreamSection, DreamSummary, write_dream
from mltgnt.memory.dream.api import read_dream_summary

PERSONA_CONTENT = textwrap.dedent("""\
    ---
    persona:
      name: persona-a
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## \u57fa\u672c\u60c5\u5831

    persona-a is a multi-legged tank-type AI robot from GHS.
""")


@pytest.fixture
def chat_dir(tmp_path: Path) -> Path:
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "persona-a.md").write_text(PERSONA_CONTENT, encoding="utf-8")
    return agents


def _memory_config(chat_dir: Path) -> MemoryConfig:
    return MemoryConfig(chat_dir=chat_dir)


def _write_dream(chat_dir: Path, persona: str, sections: list[DreamSection]) -> None:
    summary = DreamSummary(
        persona=persona,
        sections=sections,
        updated_at="2026-06-07T12:00:00+09:00",
    )
    write_dream(chat_dir / persona, summary)


def test_read_dream_summary_formats_multiple_sections(chat_dir: Path) -> None:
    _write_dream(
        chat_dir,
        "persona-a",
        [
            DreamSection(category="behavior patterns", content="Starts work as a morning person.", source_entries=2),
            DreamSection(category="preferences", content="Prefers concise explanations.", source_entries=1),
        ],
    )
    config = _memory_config(chat_dir)

    result = read_dream_summary(config, "persona-a")

    # Japanese text intentionally kept for CJK processing test
    assert result.startswith("\n\n## \u8a18\u61b6\u306e\u8981\u7d04\n\n")
    assert "### behavior patterns\nStarts work as a morning person." in result
    assert "### preferences\nPrefers concise explanations." in result


def test_read_dream_summary_returns_empty_when_no_dream_json(chat_dir: Path) -> None:
    config = _memory_config(chat_dir)
    assert read_dream_summary(config, "persona-a") == ""


def test_read_dream_summary_returns_empty_when_sections_empty(chat_dir: Path) -> None:
    _write_dream(chat_dir, "persona-a", [])
    config = _memory_config(chat_dir)
    assert read_dream_summary(config, "persona-a") == ""


def test_run_pipeline_injects_dream_summary(chat_dir: Path) -> None:
    from mltgnt.chat.pipeline import run_pipeline
    from mltgnt.persona.loader import load
    from mltgnt.persona.registry import resolve_with_alias
    from mltgnt.persona.schema import SYSTEM_DEFAULT_ENGINE, SYSTEM_DEFAULT_MODEL

    _write_dream(
        chat_dir,
        "persona-a",
        [DreamSection(category="behavior patterns", content="dream injection test", source_entries=1)],
    )
    config = _memory_config(chat_dir)
    dream_text = read_dream_summary(config, "persona-a")

    path = resolve_with_alias("persona-a", chat_dir)
    persona = load(path)
    engine = persona.fm.engine or SYSTEM_DEFAULT_ENGINE
    model = persona.fm.model or SYSTEM_DEFAULT_MODEL

    mock_result = type("R", (), {"success": True, "body": "response", "stderr": ""})()
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=mock_result) as mock_call:
        run_pipeline("test", persona, engine=engine, model=model, memory=dream_text)

    called_prompt: str = mock_call.call_args[0][0]
    # Japanese text intentionally kept for CJK processing test
    assert "## \u8a18\u61b6\u306e\u8981\u7d04" in called_prompt
    assert "dream injection test" in called_prompt
    assert f"{dream_text}\n\ntest" in called_prompt
