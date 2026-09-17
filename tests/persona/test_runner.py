"""Tests for mltgnt.persona.runner — LLM calls via ghdag.llm.call()."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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

    ## \u4fa1\u5024\u89b3

    Curious.

    ## \u53cd\u5fdc\u30d1\u30bf\u30fc\u30f3

    Answers questions.

    ## \u53e3\u8abf

    Friendly.

    ## \u30a2\u30a6\u30c8\u30d7\u30c3\u30c8\u5f62\u5f0f

    Be concise.
""")


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agents"
    d.mkdir()
    (d / "persona-a.md").write_text(PERSONA_CONTENT, encoding="utf-8")
    return d


def _make_llm_result(ok: bool = True, stdout: str = "response", stderr: str = "") -> MagicMock:
    r = MagicMock()
    r.success = ok
    r.body = stdout
    r.stderr = stderr
    return r


def test_runner_uses_ghdag_llm(persona_dir: Path) -> None:
    """run_persona_prompt must call ghdag.llm.call(), not subprocess."""
    from mltgnt.persona.runner import run_persona_prompt

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout="test response")) as mock_call:
        result = run_persona_prompt("persona-a", "hello", persona_dir=persona_dir)

    mock_call.assert_called_once()
    assert result == "test response"


def test_runner_passes_engine_and_model(persona_dir: Path) -> None:
    """engine / model must be passed correctly to ghdag.llm.call."""
    from mltgnt.persona.runner import run_persona_prompt

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result()) as mock_call:
        run_persona_prompt("persona-a", "test", persona_dir=persona_dir)

    _, kwargs = mock_call.call_args
    assert kwargs.get("engine") == "claude"
    assert kwargs.get("model") == "claude-sonnet-4-6"


def test_runner_passes_timeout(persona_dir: Path) -> None:
    """timeout must be passed to ghdag.llm.call."""
    from mltgnt.persona.runner import run_persona_prompt

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result()) as mock_call:
        run_persona_prompt("persona-a", "test", persona_dir=persona_dir, timeout=30)

    _, kwargs = mock_call.call_args
    assert kwargs.get("timeout") == 30


def test_runner_ok_false_returns_error_string(persona_dir: Path) -> None:
    """When ghdag.llm.call returns ok=False, return an error string."""
    from mltgnt.persona.runner import run_persona_prompt

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(ok=False, stderr="engine error")):
        result = run_persona_prompt("persona-a", "test", persona_dir=persona_dir)

    assert "error" in result


def test_runner_exception_returns_error_string(persona_dir: Path) -> None:
    """When ghdag.llm.call raises, return an error string."""
    from mltgnt.persona.runner import run_persona_prompt

    with patch("mltgnt.bridges.llm_adapter.call_llm", side_effect=RuntimeError("connection refused")):
        result = run_persona_prompt("persona-a", "test", persona_dir=persona_dir)

    assert "exec failed" in result


def test_run_persona_prompt_rejects_audit_writer_kwarg(persona_dir: Path) -> None:
    """audit_writer keyword argument must raise TypeError."""
    from mltgnt.persona.runner import run_persona_prompt

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(ok=True, stdout="ok")):
        with pytest.raises(TypeError):
            run_persona_prompt("persona-a", "hello", persona_dir=persona_dir, audit_writer=MagicMock())


def test_persona_runner_does_not_import_chat_module() -> None:
    """persona/runner.py must not import mltgnt.chat."""
    import inspect

    import mltgnt.persona.runner as mod

    source = inspect.getsource(mod)
    assert "mltgnt.chat" not in source
