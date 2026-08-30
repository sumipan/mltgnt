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
      name: タチコマ
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## 基本情報

    タチコマはGHSの多脚戦車型AIロボット。
""")


@pytest.fixture
def chat_dir(tmp_path: Path) -> Path:
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "タチコマ.md").write_text(PERSONA_CONTENT, encoding="utf-8")
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
        "タチコマ",
        [
            DreamSection(category="行動パターン", content="朝型で作業を始める。", source_entries=2),
            DreamSection(category="好み・傾向", content="簡潔な説明を好む。", source_entries=1),
        ],
    )
    config = _memory_config(chat_dir)

    result = read_dream_summary(config, "タチコマ")

    assert result.startswith("\n\n## 記憶の要約\n\n")
    assert "### 行動パターン\n朝型で作業を始める。" in result
    assert "### 好み・傾向\n簡潔な説明を好む。" in result


def test_read_dream_summary_returns_empty_when_no_dream_json(chat_dir: Path) -> None:
    config = _memory_config(chat_dir)
    assert read_dream_summary(config, "タチコマ") == ""


def test_read_dream_summary_returns_empty_when_sections_empty(chat_dir: Path) -> None:
    _write_dream(chat_dir, "タチコマ", [])
    config = _memory_config(chat_dir)
    assert read_dream_summary(config, "タチコマ") == ""


def test_run_pipeline_injects_dream_summary(chat_dir: Path) -> None:
    from mltgnt.chat.pipeline import run_pipeline
    from mltgnt.persona.loader import load
    from mltgnt.persona.registry import resolve_with_alias
    from mltgnt.persona.schema import SYSTEM_DEFAULT_ENGINE, SYSTEM_DEFAULT_MODEL

    _write_dream(
        chat_dir,
        "タチコマ",
        [DreamSection(category="行動パターン", content="dream注入テスト", source_entries=1)],
    )
    config = _memory_config(chat_dir)
    dream_text = read_dream_summary(config, "タチコマ")

    path = resolve_with_alias("タチコマ", chat_dir)
    persona = load(path)
    engine = persona.fm.engine or SYSTEM_DEFAULT_ENGINE
    model = persona.fm.model or SYSTEM_DEFAULT_MODEL

    mock_result = type("R", (), {"success": True, "body": "応答", "stderr": ""})()
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=mock_result) as mock_call:
        run_pipeline("テスト", persona, engine=engine, model=model, memory=dream_text)

    called_prompt: str = mock_call.call_args[0][0]
    assert "## 記憶の要約" in called_prompt
    assert "dream注入テスト" in called_prompt
    assert f"{dream_text}\n\nテスト" in called_prompt
