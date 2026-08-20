"""tests/loops/test_prompts.py — JSON 抽出・契約検証テスト。"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from mltgnt.loops import prompts


def test_extract_json_from_fence():
    text = 'Here:\n```json\n{"clear": true, "question": null, "reason": "ok", "reasoning": "r", "uncertain_flag": false}\n```'
    data = prompts.extract_json(text)
    assert data["clear"] is True


def test_extract_json_raw():
    text = 'prefix {"clear": false, "question": "Q?", "reason": "r", "reasoning": "r", "uncertain_flag": false} suffix'
    data = prompts.extract_json(text)
    assert data["question"] == "Q?"


def test_validate_clarify_rejects_clear_with_question():
    with pytest.raises(ValueError, match="clear=true"):
        prompts._validate_clarify(
            {"clear": True, "question": "x", "reason": "", "reasoning": "", "uncertain_flag": False}
        )


def test_validate_clarify_requires_question_when_not_clear():
    with pytest.raises(ValueError, match="non-empty question"):
        prompts._validate_clarify(
            {"clear": False, "question": "", "reason": "", "reasoning": "", "uncertain_flag": False}
        )


def test_validate_decompose_empty_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        prompts._validate_decompose({"subtasks": [], "reasoning": "", "uncertain_flag": False}, max_subtasks=5)


def test_validate_evaluate_next_focus_required():
    with pytest.raises(ValueError, match="next_focus"):
        prompts._validate_evaluate(
            {"achieved": False, "score": 50, "summary": "s", "next_focus": "", "reasoning": "", "uncertain_flag": False}
        )


def test_failed_json_retry_preserves_failure_trace():
    with patch("mltgnt.loops.prompts.call_llm", side_effect=["not json", "still not json"]):
        with pytest.raises(prompts.LlmCallError) as exc_info:
            prompts.run_clarify("prompt", engine="claude", model="model")

    trace = exc_info.value.trace
    assert trace.input == "prompt"
    assert trace.raw_output == "still not json"
    assert trace.error
    assert trace.metadata["retry"] is True
    assert trace.metadata["token_usage_reason"]
