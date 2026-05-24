"""Tests for mltgnt.chat.pipeline.run_chat"""
from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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

    ## 価値観

    好奇心旺盛。

    ## 反応パターン

    質問に答える。

    ## 口調

    フレンドリー。

    ## アウトプット形式

    簡潔に。
""")


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agents"
    d.mkdir()
    (d / "タチコマ.md").write_text(PERSONA_CONTENT, encoding="utf-8")
    return d


def _make_llm_result(ok: bool = True, stdout: str = "応答", stderr: str = "") -> MagicMock:
    r = MagicMock()
    r.ok = ok
    r.stdout = stdout
    r.stderr = stderr
    return r


def test_run_chat_returns_chat_output(persona_dir: Path) -> None:
    """run_chat は ChatOutput を返すこと。"""
    from mltgnt.chat.pipeline import run_chat

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout="テスト応答")):
        out = run_chat("こんにちは", "タチコマ", persona_dir)

    from mltgnt.chat.models import ChatOutput
    assert isinstance(out, ChatOutput)


def test_run_chat_content_has_llm_response(persona_dir: Path) -> None:
    """ChatOutput.content に LLM 応答テキストが格納されること。"""
    from mltgnt.chat.pipeline import run_chat

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout="テスト応答")):
        out = run_chat("こんにちは", "タチコマ", persona_dir)

    assert out.content == "テスト応答"


def test_run_chat_persona_name_matches(persona_dir: Path) -> None:
    """ChatOutput.persona_name が渡したペルソナ名と一致すること。"""
    from mltgnt.chat.pipeline import run_chat

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result()):
        out = run_chat("テスト", "タチコマ", persona_dir)

    assert out.persona_name == "タチコマ"


def test_run_chat_timestamp_is_asia_tokyo(persona_dir: Path) -> None:
    """ChatOutput.timestamp が Asia/Tokyo タイムゾーンの datetime であること。"""
    from mltgnt.chat.pipeline import run_chat

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result()):
        out = run_chat("テスト", "タチコマ", persona_dir)

    assert isinstance(out.timestamp, datetime)
    assert out.timestamp.tzinfo is not None
    assert out.timestamp.utcoffset().total_seconds() == 9 * 3600


def test_run_chat_memory_prepended(persona_dir: Path) -> None:
    """memory が非 None の場合、プロンプト先頭に付加されること。"""
    from mltgnt.chat.pipeline import run_chat

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result()) as mock_call:
        run_chat("テスト", "タチコマ", persona_dir, memory="メモリ内容")

    called_prompt: str = mock_call.call_args[0][0]
    assert "メモリ内容\n\n" in called_prompt


def test_run_chat_ok_false_returns_error_content(persona_dir: Path) -> None:
    """LLM が ok=False を返した場合、content に "（エラー: ...）" が含まれること。"""
    from mltgnt.chat.pipeline import run_chat

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(ok=False, stderr="engine error")):
        out = run_chat("テスト", "タチコマ", persona_dir)

    assert "エラー" in out.content
    assert "engine error" in out.content


def test_run_chat_exception_returns_error_content(persona_dir: Path) -> None:
    """LLM が RuntimeError を送出した場合、content に "（実行失敗: ...）" が含まれ例外は送出されないこと。"""
    from mltgnt.chat.pipeline import run_chat

    with patch("mltgnt.bridges.llm_adapter.call_llm", side_effect=RuntimeError("connection refused")):
        out = run_chat("テスト", "タチコマ", persona_dir)

    assert "実行失敗" in out.content
    assert "connection refused" in out.content


def test_run_chat_audit_writer_called(persona_dir: Path) -> None:
    """audit_writer が1回呼ばれ、必須キーが含まれること。"""
    from mltgnt.chat.pipeline import run_chat

    mock_writer = MagicMock()
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(ok=True, stdout="応答")):
        run_chat("テスト", "タチコマ", persona_dir, audit_writer=mock_writer)

    mock_writer.assert_called_once()
    audit_dict = mock_writer.call_args[0][0]
    assert set(audit_dict.keys()) >= {"source", "engine", "model", "ok", "timestamp"}


def test_run_chat_audit_writer_exception_does_not_affect_result(persona_dir: Path) -> None:
    """audit_writer が例外を送出しても run_chat の戻り値に影響しないこと。"""
    from mltgnt.chat.pipeline import run_chat

    def failing_writer(_: dict) -> None:
        raise ValueError("audit write error")

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(ok=True, stdout="正常応答")):
        out = run_chat("テスト", "タチコマ", persona_dir, audit_writer=failing_writer)

    assert out.content == "正常応答"


def test_run_chat_no_ghdag_import_in_pipeline() -> None:
    """chat/pipeline.py 内に ghdag の直接 import がないこと（L3→L0 依存排除）。"""
    import inspect
    import mltgnt.chat.pipeline as mod

    source = inspect.getsource(mod)
    assert "from ghdag" not in source
    assert "import ghdag" not in source


def test_bridges_llm_adapter_importable() -> None:
    """from mltgnt.bridges.llm_adapter import call_llm が成功すること。"""
    from mltgnt.bridges.llm_adapter import call_llm  # noqa: F401


def test_chat_pipeline_importable_via_init() -> None:
    """from mltgnt.chat import run_chat が成功すること。"""
    from mltgnt.chat import run_chat  # noqa: F401
