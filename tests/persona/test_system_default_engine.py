"""Tests for mltgnt.persona.schema.system_default_engine and its call sites (#3809)."""
from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.persona.result_format import format_result_for_persona
from mltgnt.persona.schema import SYSTEM_DEFAULT_ENGINE, system_default_engine

ENV = "MLTGNT_DEFAULT_ENGINE"
ENGINES = ["claude", "cursor", "codex"]
LOGGER = logging.getLogger("test_system_default_engine")


def _persona_md(name: str, engine: str | None) -> str:
    ops = f"ops:\n  engine: {engine}\n" if engine else ""
    body = textwrap.dedent(f"""\
        ## Basic information

        {name} is a synthetic test persona.

        ## Tone

        Plain.
    """)
    return f"---\npersona:\n  name: {name}\n{ops}---\n\n{body}"


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agents"
    d.mkdir()
    (d / "persona-noengine.md").write_text(_persona_md("persona-noengine", None), encoding="utf-8")
    (d / "persona-cursor.md").write_text(_persona_md("persona-cursor", "cursor"), encoding="utf-8")
    return d


def _llm_result(body: str = "ok") -> MagicMock:
    r = MagicMock()
    r.success = True
    r.body = body
    r.stderr = ""
    r.returncode = 0
    return r


# ---------------------------------------------------------------------------
# system_default_engine()
# ---------------------------------------------------------------------------


def test_constant_is_unchanged() -> None:
    assert SYSTEM_DEFAULT_ENGINE == "claude"


def test_unset_returns_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    assert system_default_engine() == "claude"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", "claude"),
        ("  ", "claude"),
        ("cursor", "cursor"),
        (" codex ", "codex"),
        ("claude", "claude"),
        ("gemini", "gemini"),
    ],
)
def test_env_values(monkeypatch: pytest.MonkeyPatch, value: str, expected: str) -> None:
    monkeypatch.setenv(ENV, value)
    assert system_default_engine() == expected


@pytest.mark.parametrize("value", ["foo", "Cursor"])
def test_invalid_value_raises(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match=repr(value)):
        system_default_engine()


def test_reads_env_on_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV, "cursor")
    assert system_default_engine() == "cursor"
    monkeypatch.setenv(ENV, "codex")
    assert system_default_engine() == "codex"


# ---------------------------------------------------------------------------
# run_persona_prompt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_runner_uses_env_default(
    monkeypatch: pytest.MonkeyPatch, persona_dir: Path, engine: str
) -> None:
    from mltgnt.persona.runner import run_persona_prompt

    monkeypatch.setenv(ENV, engine)
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_llm_result()) as mock_call:
        run_persona_prompt("persona-noengine", "hi", persona_dir=persona_dir)
    assert mock_call.call_args.kwargs["engine"] == engine


@pytest.mark.parametrize("value", ["foo", "Cursor"])
def test_runner_invalid_env_propagates(
    monkeypatch: pytest.MonkeyPatch, persona_dir: Path, value: str
) -> None:
    from mltgnt.persona.runner import run_persona_prompt

    monkeypatch.setenv(ENV, value)
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_llm_result()) as mock_call:
        with pytest.raises(ValueError, match=repr(value)):
            run_persona_prompt("persona-noengine", "hi", persona_dir=persona_dir)
    mock_call.assert_not_called()


@pytest.mark.parametrize("value", ["claude", "foo"])
def test_runner_explicit_engine_wins(
    monkeypatch: pytest.MonkeyPatch, persona_dir: Path, value: str
) -> None:
    from mltgnt.persona.runner import run_persona_prompt

    monkeypatch.setenv(ENV, value)
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_llm_result()) as mock_call:
        run_persona_prompt("persona-cursor", "hi", persona_dir=persona_dir)
    assert mock_call.call_args.kwargs["engine"] == "cursor"


# ---------------------------------------------------------------------------
# format_result_for_persona
# ---------------------------------------------------------------------------


def _fake_llm_call(calls: list[dict]):
    def _call(prompt: str, **kwargs):
        calls.append(kwargs)
        return _llm_result("formatted")

    return _call


@pytest.mark.parametrize("engine", ENGINES)
def test_format_uses_env_default(monkeypatch: pytest.MonkeyPatch, engine: str) -> None:
    monkeypatch.setenv(ENV, engine)
    calls: list[dict] = []
    out = format_result_for_persona(
        "body", prompt_header="h", llm_call=_fake_llm_call(calls), logger=LOGGER, engine=""
    )
    assert out == "formatted"
    assert calls[0]["engine"] == engine


@pytest.mark.parametrize("value", ["foo", "Cursor"])
def test_format_invalid_env_propagates(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ENV, value)
    calls: list[dict] = []
    with pytest.raises(ValueError, match=repr(value)):
        format_result_for_persona(
            "body", prompt_header="h", llm_call=_fake_llm_call(calls), logger=LOGGER, engine=""
        )
    assert calls == []


@pytest.mark.parametrize("value", ["claude", "foo"])
def test_format_explicit_engine_wins(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ENV, value)
    calls: list[dict] = []
    format_result_for_persona(
        "body", prompt_header="h", llm_call=_fake_llm_call(calls), logger=LOGGER, engine="codex"
    )
    assert calls[0]["engine"] == "codex"
