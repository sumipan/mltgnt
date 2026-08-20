"""tests/loops/test_prompts.py — JSON 抽出・契約検証テスト。"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from mltgnt.loops import prompts
from tests.loops.fakes import make_llm_result

_CLARIFY_OK = (
    '{"clear": true, "question": null, "reason": "ok", '
    '"reasoning": "r", "uncertain_flag": false}'
)
_DECOMPOSE_OK = (
    '{"subtasks": [{"id": "s1", "title": "T", "kind": "auto", "prompt": "p"}], '
    '"reasoning": "r", "uncertain_flag": false}'
)
_EVALUATE_OK = (
    '{"achieved": true, "score": 90, "summary": "done", "next_focus": "", '
    '"reasoning": "r", "uncertain_flag": false}'
)


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


def test_run_clarify_parses_llm_result_stdout():
    with patch(
        "mltgnt.loops.prompts.call_llm",
        return_value=make_llm_result(stdout=_CLARIFY_OK),
    ):
        resp, trace = prompts.run_clarify("prompt", engine="claude", model="model")

    assert resp.clear is True
    assert resp.question is None
    assert trace.raw_output == _CLARIFY_OK
    assert trace.error is None
    assert trace.metadata["retry"] is False


def test_run_decompose_parses_llm_result_stdout():
    with patch(
        "mltgnt.loops.prompts.call_llm",
        return_value=make_llm_result(stdout=_DECOMPOSE_OK),
    ):
        resp, _trace = prompts.run_decompose(
            "prompt", engine="claude", model="model", max_subtasks=5
        )

    assert len(resp.subtasks) == 1
    assert resp.subtasks[0].id == "s1"


def test_run_evaluate_parses_llm_result_stdout():
    with patch(
        "mltgnt.loops.prompts.call_llm",
        return_value=make_llm_result(stdout=_EVALUATE_OK),
    ):
        resp, _trace = prompts.run_evaluate("prompt", engine="claude", model="model")

    assert resp.achieved is True
    assert resp.score == 90


def test_failed_json_retry_preserves_failure_trace():
    with patch(
        "mltgnt.loops.prompts.call_llm",
        side_effect=[
            make_llm_result(stdout="not json"),
            make_llm_result(stdout="still not json"),
        ],
    ):
        with pytest.raises(prompts.LlmCallError) as exc_info:
            prompts.run_clarify("prompt", engine="claude", model="model")

    trace = exc_info.value.trace
    assert trace.input == "prompt"
    assert trace.raw_output == "still not json"
    assert trace.error
    assert trace.metadata["retry"] is True
    assert trace.metadata["token_usage_reason"]


def test_ok_false_retries_then_raises_llm_call_error():
    with patch(
        "mltgnt.loops.prompts.call_llm",
        side_effect=[
            make_llm_result(ok=False, stderr="engine boom"),
            make_llm_result(ok=False, stderr="engine boom again"),
        ],
    ) as mock_call:
        with pytest.raises(prompts.LlmCallError) as exc_info:
            prompts.run_clarify("prompt", engine="claude", model="model")

    assert mock_call.call_count == 2
    assert "engine boom again" in str(exc_info.value)
    assert exc_info.value.trace.metadata["retry"] is True


def test_ok_false_then_ok_succeeds_on_retry():
    with patch(
        "mltgnt.loops.prompts.call_llm",
        side_effect=[
            make_llm_result(ok=False, stderr="transient"),
            make_llm_result(stdout=_CLARIFY_OK),
        ],
    ):
        resp, trace = prompts.run_clarify("prompt", engine="claude", model="model")

    assert resp.clear is True
    assert trace.metadata["retry"] is True