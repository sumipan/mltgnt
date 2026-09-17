"""mltgnt.persona.result_format — pin formatting skeleton with 3-engine fixtures (#3318)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mltgnt.persona.result_format import format_result_for_persona

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "format_result"

_ENGINES = ("claude", "cursor", "codex")
_CASES = ("success", "failure", "empty")


@dataclass(frozen=True)
class _FakeLLMResult:
    body: str
    success: bool
    returncode: int = 0
    stderr: str = ""


def _load_fixture(engine: str, case: str) -> dict[str, Any]:
    path = _FIXTURE_DIR / f"{engine}_{case}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("case", _CASES)
def test_format_result_for_persona_matches_fixture_expectation(
    engine: str, case: str
) -> None:
    fx = _load_fixture(engine, case)
    calls: list[dict[str, Any]] = []

    def _llm_call(
        _prompt: str,
        *,
        stdin_text: str | None = None,
        engine: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> _FakeLLMResult:
        calls.append(
            {
                "stdin_text": stdin_text,
                "engine": engine,
                "model": model,
                "timeout": timeout,
            }
        )
        return _FakeLLMResult(
            body=fx["llm_body"],
            success=fx["success"],
            returncode=fx["returncode"],
            stderr=fx["stderr"],
        )

    logger = SimpleNamespace(warning=lambda *_a, **_k: None)
    prompt_header = "[Persona]\ntest tone\n\n--- input text ---\n"
    result = format_result_for_persona(
        fx["raw_body"],
        prompt_header=prompt_header,
        llm_call=_llm_call,
        logger=logger,
        engine=engine,
        model=f"{engine}-model",
    )

    if case == "success":
        assert result == fx["llm_body"].strip()
        assert calls and calls[0]["engine"] == engine
        assert fx["raw_body"] in (calls[0]["stdin_text"] or "")
        if engine == "codex":
            assert "CLI_EVENT_LEAK" not in (result or "")
            assert "thread.started" not in (result or "")
    else:
        assert result is None


def test_format_result_for_persona_uses_injected_postprocess() -> None:
    def _llm_call(*_a, **_k) -> _FakeLLMResult:
        return _FakeLLMResult(body="**bold** body text", success=True)

    def _post(text: str) -> str:
        return text.replace("**", "")

    logger = SimpleNamespace(warning=lambda *_a, **_k: None)
    out = format_result_for_persona(
        "draft",
        prompt_header="HDR\n",
        llm_call=_llm_call,
        logger=logger,
        engine="claude",
        postprocess=_post,
    )
    assert out == "bold body text"


def test_format_result_for_persona_empty_raw_returns_none() -> None:
    logger = SimpleNamespace(warning=lambda *_a, **_k: None)
    assert (
        format_result_for_persona(
            "  ",
            prompt_header="HDR\n",
            llm_call=lambda *_a, **_k: _FakeLLMResult("x", True),
            logger=logger,
            engine="claude",
        )
        is None
    )


def test_format_result_source_has_no_slack_mrkdwn() -> None:
    src = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mltgnt"
        / "persona"
        / "result_format.py"
    ).read_text(encoding="utf-8")
    assert "markdown_to_slack_mrkdwn" not in src
    assert "slack_sdk" not in src
    assert "ghdag" not in src
