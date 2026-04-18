"""Tests for mltgnt.chat.pipeline (AC3)."""
from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from mltgnt.chat import ChatConfig, ChatPipeline
from mltgnt.chat.models import ChatInput, ChatOutput


PERSONA_CONTENT = textwrap.dedent("""\
    ---
    persona:
      name: テストペルソナ
    ops:
      engine: claude
    ---

    ## 基本情報
    テスト用ペルソナ。

    ## 価値観
    正確さ。

    ## 反応パターン
    質問に答える。

    ## 口調
    丁寧。

    ## アウトプット形式
    簡潔に。
""")


@pytest.fixture
def agents_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agents"
    d.mkdir()
    (d / "テストペルソナ.md").write_text(PERSONA_CONTENT, encoding="utf-8")
    return d


@pytest.fixture
def mock_subprocess_run():
    """subprocess.run をモックして "mock response" を返す。"""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "mock response"
        mock_run.return_value.stderr = ""
        mock_run.return_value.returncode = 0
        yield mock_run


# ---------------------------------------------------------------------------
# AC3: ChatPipeline
# ---------------------------------------------------------------------------


def test_run_pipeline_basic(agents_dir: Path, mock_subprocess_run) -> None:
    """AC3 正常系: run_pipeline が ChatOutput を返す。"""
    config = ChatConfig(persona_dir=agents_dir)
    pipeline = ChatPipeline(config)
    inp = ChatInput(
        source="test",
        session_key="session-1",
        messages=[{"role": "user", "content": "こんにちは"}],
        persona_name="テストペルソナ",
    )
    out = pipeline.run_pipeline(inp)
    assert isinstance(out, ChatOutput)
    assert out.content == "mock response"
    assert out.persona_name == "テストペルソナ"
    assert out.session_key == "session-1"
    assert isinstance(out.timestamp, datetime)


def test_run_pipeline_no_memory(agents_dir: Path, mock_subprocess_run) -> None:
    """AC3 正常系: memory_dir=None のときメモリ注入なしで動作する。"""
    config = ChatConfig(persona_dir=agents_dir, memory_dir=None)
    pipeline = ChatPipeline(config)
    inp = ChatInput(
        source="test",
        session_key="s2",
        messages=[{"role": "user", "content": "テスト"}],
        persona_name="テストペルソナ",
    )
    out = pipeline.run_pipeline(inp)
    assert out.content == "mock response"


def test_run_pipeline_persona_not_found(agents_dir: Path) -> None:
    """AC3 異常系: 存在しないペルソナ名は FileNotFoundError。"""
    config = ChatConfig(persona_dir=agents_dir)
    pipeline = ChatPipeline(config)
    inp = ChatInput(
        source="test",
        session_key="s3",
        messages=[{"role": "user", "content": "hello"}],
        persona_name="存在しないペルソナ",
    )
    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline(inp)


def test_chat_config_fields(agents_dir: Path) -> None:
    """ChatConfig の各フィールドが正しく設定される。"""
    config = ChatConfig(persona_dir=agents_dir, memory_dir=Path("/tmp/memory"))
    assert config.persona_dir == agents_dir
    assert config.memory_dir == Path("/tmp/memory")


def test_chat_config_default_memory_dir(agents_dir: Path) -> None:
    """ChatConfig の memory_dir デフォルトは None。"""
    config = ChatConfig(persona_dir=agents_dir)
    assert config.memory_dir is None
