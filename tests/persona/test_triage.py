"""Tests for extract_json_object, run_triage_once, run_slack_triage."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, call, patch

import pytest

from mltgnt.persona.triage import (
    DEFAULT_TIMEOUT_SEC,
    GEMINI_TIMEOUT_SEC,
    extract_json_object,
    run_slack_triage,
    run_triage_once,
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
# run_triage_once
# ---------------------------------------------------------------------------


def test_run_triage_once_success():
    mock_logger = MagicMock()
    completed = MagicMock()
    completed.stdout = '{"mode":"direct"}'
    completed.returncode = 0
    with patch("subprocess.run", return_value=completed):
        result = run_triage_once(["gemini", "-p", "prompt"], "prompt", mock_logger, timeout=25)
    assert result == {"mode": "direct"}


def test_run_triage_once_timeout():
    import subprocess

    mock_logger = MagicMock()
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="gemini", timeout=25)):
        result = run_triage_once(["gemini", "-p", "prompt"], "prompt", mock_logger, timeout=25)
    assert result is None


def test_run_triage_once_invalid_json():
    mock_logger = MagicMock()
    completed = MagicMock()
    completed.stdout = "not json"
    completed.returncode = 0
    with patch("subprocess.run", return_value=completed):
        result = run_triage_once(["gemini", "-p", "prompt"], "prompt", mock_logger, timeout=25)
    assert result is None


def test_run_triage_once_nonzero_exit():
    mock_logger = MagicMock()
    completed = MagicMock()
    completed.stdout = '{"mode":"delegate"}'
    completed.returncode = 1
    completed.stderr = "some error"
    with patch("subprocess.run", return_value=completed):
        result = run_triage_once(["claude", "-p", "prompt"], "prompt", mock_logger, timeout=60)
    assert result == {"mode": "delegate"}


# ---------------------------------------------------------------------------
# run_slack_triage
# ---------------------------------------------------------------------------


def test_run_slack_triage_first_try_ok():
    mock_logger = MagicMock()
    expected = {"mode": "direct", "reply": "hi"}
    with patch("mltgnt.persona.triage.run_triage_once", return_value=expected) as mock_once:
        result = run_slack_triage("hello", "# profile", mock_logger)
    assert result == expected
    assert mock_once.call_count == 1


def test_run_slack_triage_retry_ok():
    mock_logger = MagicMock()
    expected = {"mode": "delegate"}
    with patch("mltgnt.persona.triage.run_triage_once", side_effect=[None, expected]) as mock_once:
        result = run_slack_triage("hello", "# profile", mock_logger)
    assert result == expected
    assert mock_once.call_count == 2


def test_run_slack_triage_all_fail():
    mock_logger = MagicMock()
    with patch("mltgnt.persona.triage.run_triage_once", return_value=None) as mock_once:
        result = run_slack_triage("hello", "# profile", mock_logger)
    assert result is None
    assert mock_once.call_count == 2


def test_run_slack_triage_gemini_timeout():
    mock_logger = MagicMock()
    captured_kwargs = []

    def capture(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return {"mode": "direct"}

    with patch("mltgnt.persona.triage.run_triage_once", side_effect=capture):
        run_slack_triage("hello", None, mock_logger, engine="gemini")

    assert captured_kwargs[0]["timeout"] == GEMINI_TIMEOUT_SEC


def test_run_slack_triage_default_timeout():
    mock_logger = MagicMock()
    captured_kwargs = []

    def capture(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return {"mode": "direct"}

    with patch("mltgnt.persona.triage.run_triage_once", side_effect=capture):
        run_slack_triage("hello", None, mock_logger, engine="claude")

    assert captured_kwargs[0]["timeout"] == DEFAULT_TIMEOUT_SEC
