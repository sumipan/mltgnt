"""Tests for extract_json_object, run_slack_triage."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch


from mltgnt.persona.triage import (
    DEFAULT_TIMEOUT_SEC,
    GEMINI_TIMEOUT_SEC,
    extract_json_object,
    run_slack_triage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# extract_json_object
# ---------------------------------------------------------------------------


def test_extract_json_object_empty():
    assert extract_json_object("") is None


def test_extract_json_object_valid():
    assert extract_json_object('{"mode": "direct"}') == {"mode": "direct"}


def test_extract_json_object_with_fence():
    assert extract_json_object('```json\n{"mode": "delegate"}\n```') == {"mode": "delegate"}


def test_extract_json_object_invalid():
    assert extract_json_object("{not json}") is None


def test_extract_json_object_surrounded():
    assert extract_json_object('prefix {"k": "v"} suffix') == {"k": "v"}


# ---------------------------------------------------------------------------
# run_slack_triage — ghdag.llm.call 経由
# ---------------------------------------------------------------------------


def _make_llm_result(ok: bool = True, stdout: str = "", stderr: str = "") -> MagicMock:
    r = MagicMock()
    r.ok = ok
    r.stdout = stdout
    r.stderr = stderr
    return r


def test_triage_uses_ghdag_llm():
    """persona.triage は ghdag.llm.call() 経由で LLM を呼ぶこと。"""
    mock_logger = MagicMock()
    with patch("ghdag.llm.call", return_value=_make_llm_result(stdout='{"mode":"direct","reply":"hi"}')) as mock_call:
        result = run_slack_triage("hello", "# profile", mock_logger)
    mock_call.assert_called_once()
    assert result == {"mode": "direct", "reply": "hi"}


def test_run_slack_triage_first_try_ok():
    mock_logger = MagicMock()
    expected = {"mode": "direct", "reply": "hi"}
    with patch("ghdag.llm.call", return_value=_make_llm_result(stdout='{"mode":"direct","reply":"hi"}')) as mock_call:
        result = run_slack_triage("hello", "# profile", mock_logger)
    assert result == expected
    assert mock_call.call_count == 1


def test_run_slack_triage_retry_on_invalid_json():
    """1回目が不正JSONの場合、リトライして正常結果を返すこと。"""
    mock_logger = MagicMock()
    expected = {"mode": "delegate"}
    results = [
        _make_llm_result(stdout="not json"),
        _make_llm_result(stdout='{"mode":"delegate"}'),
    ]
    with patch("ghdag.llm.call", side_effect=results) as mock_call:
        result = run_slack_triage("hello", "# profile", mock_logger)
    assert result == expected
    assert mock_call.call_count == 2


def test_run_slack_triage_all_fail():
    """両回とも不正JSONなら None を返すこと。"""
    mock_logger = MagicMock()
    with patch("ghdag.llm.call", return_value=_make_llm_result(stdout="not json")) as mock_call:
        result = run_slack_triage("hello", "# profile", mock_logger)
    assert result is None
    assert mock_call.call_count == 2


def test_run_slack_triage_gemini_timeout():
    """engine=gemini のとき GEMINI_TIMEOUT_SEC が渡されること。"""
    mock_logger = MagicMock()
    captured_kwargs = []

    def capture(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return _make_llm_result(stdout='{"mode":"direct"}')

    with patch("ghdag.llm.call", side_effect=capture):
        run_slack_triage("hello", None, mock_logger, engine="gemini")

    assert captured_kwargs[0]["timeout"] == GEMINI_TIMEOUT_SEC


def test_run_slack_triage_default_timeout():
    """engine=claude のとき DEFAULT_TIMEOUT_SEC が渡されること。"""
    mock_logger = MagicMock()
    captured_kwargs = []

    def capture(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return _make_llm_result(stdout='{"mode":"direct"}')

    with patch("ghdag.llm.call", side_effect=capture):
        run_slack_triage("hello", None, mock_logger, engine="claude")

    assert captured_kwargs[0]["timeout"] == DEFAULT_TIMEOUT_SEC


def test_run_slack_triage_exception_returns_none():
    """ghdag.llm.call が例外を投げた場合、None を返すこと。"""
    mock_logger = MagicMock()
    with patch("ghdag.llm.call", side_effect=RuntimeError("connection error")):
        result = run_slack_triage("hello", "# profile", mock_logger)
    assert result is None
