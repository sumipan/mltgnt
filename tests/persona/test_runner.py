"""Tests for mltgnt.persona.runner — ghdag.llm.call() 経由の LLM 呼び出し。"""
from __future__ import annotations

import textwrap
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


def test_runner_uses_ghdag_llm(persona_dir: Path) -> None:
    """run_persona_prompt は subprocess ではなく ghdag.llm.call() を呼ぶこと。"""
    from mltgnt.persona.runner import run_persona_prompt

    with patch("ghdag.llm.call", return_value=_make_llm_result(stdout="テスト応答")) as mock_call:
        result = run_persona_prompt("タチコマ", "こんにちは", persona_dir=persona_dir)

    mock_call.assert_called_once()
    assert result == "テスト応答"


def test_runner_passes_engine_and_model(persona_dir: Path) -> None:
    """engine / model が ghdag.llm.call に正しく渡されること。"""
    from mltgnt.persona.runner import run_persona_prompt

    with patch("ghdag.llm.call", return_value=_make_llm_result()) as mock_call:
        run_persona_prompt("タチコマ", "テスト", persona_dir=persona_dir)

    _, kwargs = mock_call.call_args
    assert kwargs.get("engine") == "claude"
    assert kwargs.get("model") == "claude-sonnet-4-6"


def test_runner_passes_timeout(persona_dir: Path) -> None:
    """timeout が ghdag.llm.call に渡されること。"""
    from mltgnt.persona.runner import run_persona_prompt

    with patch("ghdag.llm.call", return_value=_make_llm_result()) as mock_call:
        run_persona_prompt("タチコマ", "テスト", persona_dir=persona_dir, timeout=30)

    _, kwargs = mock_call.call_args
    assert kwargs.get("timeout") == 30


def test_runner_ok_false_returns_error_string(persona_dir: Path) -> None:
    """ghdag.llm.call が ok=False を返した場合、エラー文字列を返すこと。"""
    from mltgnt.persona.runner import run_persona_prompt

    with patch("ghdag.llm.call", return_value=_make_llm_result(ok=False, stderr="engine error")):
        result = run_persona_prompt("タチコマ", "テスト", persona_dir=persona_dir)

    assert "エラー" in result


def test_runner_exception_returns_error_string(persona_dir: Path) -> None:
    """ghdag.llm.call が例外を投げた場合、エラー文字列を返すこと。"""
    from mltgnt.persona.runner import run_persona_prompt

    with patch("ghdag.llm.call", side_effect=RuntimeError("connection refused")):
        result = run_persona_prompt("タチコマ", "テスト", persona_dir=persona_dir)

    assert "実行失敗" in result
