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


def test_decompose_instruction_includes_deliverable_contract():
    text = prompts.build_decompose_instruction("body", iteration=1, max_subtasks=3)
    assert "deliverable.md" in text
    assert "Do not create new draft" in text


def test_evaluate_instruction_includes_deliverable_excerpt():
    text = prompts.build_evaluate_instruction(
        "body",
        results_summary="- s1: ok",
        iteration=2,
        max_iterations=5,
        deliverable_excerpt="integrated draft",
    )
    assert "integrated draft" in text
    assert "- s1: ok" in text


def test_build_auto_subtask_prompt_contract():
    text = prompts.build_auto_subtask_prompt(
        "do work",
        deliverable_path="/tmp/state/loop1/deliverable.md",
        deliverable_excerpt="current body",
    )
    assert "/tmp/state/loop1/deliverable.md" in text
    assert "Edit this file directly" in text
    assert "Do not create new deliverable" in text
    assert "3-5 line" in text
    assert "current body" in text


def test_validate_decompose_accepts_watch_and_defaults():
    resp = prompts.validate_decompose_payload(
        {
            "subtasks": [
                {
                    "id": "w1",
                    "title": "Wait",
                    "kind": "watch",
                    "condition": {"type": "path_exists", "path": "a"},
                },
                {"id": "a1", "title": "Do", "kind": "auto", "prompt": "work"},
            ],
            "reasoning": "",
            "uncertain_flag": False,
        },
        max_subtasks=5,
    )
    assert resp.subtasks[0].kind == "watch"
    assert resp.subtasks[0].timeout_sec == 14400
    assert resp.subtasks[0].poll_interval_sec == 60
    assert resp.subtasks[0].depends == ()
    assert resp.subtasks[1].depends == ("w1",)


def test_validate_decompose_rejects_watch_without_condition_and_bounds():
    with pytest.raises(ValueError, match="condition"):
        prompts.validate_decompose_payload(
            {"subtasks": [{"id": "w1", "kind": "watch", "title": "w"}], "reasoning": "", "uncertain_flag": False},
            max_subtasks=5,
        )
    for timeout in (59, 86401):
        with pytest.raises(ValueError, match="timeout_sec"):
            prompts.validate_decompose_payload(
                {
                    "subtasks": [{
                        "id": "w1", "kind": "watch", "title": "w",
                        "condition": {"type": "path_exists", "path": "a"},
                        "timeout_sec": timeout,
                    }],
                    "reasoning": "",
                    "uncertain_flag": False,
                },
                max_subtasks=5,
            )
    for poll in (4, 3601):
        with pytest.raises(ValueError, match="poll_interval_sec"):
            prompts.validate_decompose_payload(
                {
                    "subtasks": [{
                        "id": "w1", "kind": "watch", "title": "w",
                        "condition": {"type": "path_exists", "path": "a"},
                        "poll_interval_sec": poll,
                    }],
                    "reasoning": "",
                    "uncertain_flag": False,
                },
                max_subtasks=5,
            )


def test_validate_decompose_rejects_depends_errors_and_six():
    with pytest.raises(ValueError, match="unknown depends"):
        prompts.validate_decompose_payload(
            {
                "subtasks": [
                    {"id": "a1", "kind": "auto", "title": "a", "prompt": "p", "depends": ["missing"]},
                ],
                "reasoning": "",
                "uncertain_flag": False,
            },
            max_subtasks=5,
        )
    with pytest.raises(ValueError, match="self-dependency"):
        prompts.validate_decompose_payload(
            {
                "subtasks": [
                    {"id": "a1", "kind": "auto", "title": "a", "prompt": "p", "depends": ["a1"]},
                ],
                "reasoning": "",
                "uncertain_flag": False,
            },
            max_subtasks=5,
        )
    with pytest.raises(ValueError, match="circular"):
        prompts.validate_decompose_payload(
            {
                "subtasks": [
                    {"id": "a1", "kind": "auto", "title": "a", "prompt": "p", "depends": ["a2"]},
                    {"id": "a2", "kind": "auto", "title": "b", "prompt": "p", "depends": ["a1"]},
                ],
                "reasoning": "",
                "uncertain_flag": False,
            },
            max_subtasks=5,
        )
    with pytest.raises(ValueError, match="too many"):
        prompts.validate_decompose_payload(
            {
                "subtasks": [
                    {"id": f"a{i}", "kind": "auto", "title": "t", "prompt": "p"}
                    for i in range(6)
                ],
                "reasoning": "",
                "uncertain_flag": False,
            },
            max_subtasks=5,
        )


def test_validate_replan_requires_keep_running_success_and_rejects_dupes():
    with pytest.raises(ValueError, match="must be kept"):
        prompts.validate_replan_payload(
            {"keep": [], "add": [{"id": "n1", "kind": "auto", "title": "n", "prompt": "p"}], "reason": "r", "reasoning": "", "uncertain_flag": False},
            existing_ids={"s1", "s2"},
            required_keep={"s1"},
            max_subtasks=5,
        )
    with pytest.raises(ValueError, match="duplicate"):
        prompts.validate_replan_payload(
            {
                "keep": ["s1"],
                "add": [{"id": "s1", "kind": "auto", "title": "n", "prompt": "p"}],
                "reason": "r",
                "reasoning": "",
                "uncertain_flag": False,
            },
            existing_ids={"s1"},
            required_keep={"s1"},
            max_subtasks=5,
        )
    ok = prompts.validate_replan_payload(
        {
            "keep": ["s1"],
            "add": [{"id": "a2", "kind": "auto", "title": "n", "prompt": "p", "depends": []}],
            "reason": "fix",
            "reasoning": "",
            "uncertain_flag": False,
        },
        existing_ids={"s1", "w1"},
        required_keep={"s1"},
        max_subtasks=5,
    )
    assert ok.keep == ("s1",)
    assert ok.add[0].id == "a2"


_COMMENT_CLASSIFY_OK = (
    '{"intent": "question", "reason": "ask", "reasoning": "r", "uncertain_flag": false}'
)
_COMMENT_REPLY_OK = (
    '{"reply": "はい、進んでいます", "reasoning": "r", "uncertain_flag": false}'
)


def test_validate_comment_classify_accepts_known_intents():
    for intent in ("status", "instruction", "question", "chitchat"):
        resp = prompts.validate_comment_classify_payload(
            {"intent": intent, "reason": "", "reasoning": "", "uncertain_flag": False}
        )
        assert resp.intent == intent


def test_validate_comment_classify_rejects_unknown_intent():
    with pytest.raises(ValueError, match="intent"):
        prompts.validate_comment_classify_payload(
            {"intent": "other", "reason": "", "reasoning": "", "uncertain_flag": False}
        )


def test_validate_comment_reply_rejects_empty():
    with pytest.raises(ValueError, match="reply"):
        prompts.validate_comment_reply_payload(
            {"reply": "  ", "reasoning": "", "uncertain_flag": False}
        )


def test_run_classify_comment_parses_llm_result():
    with patch(
        "mltgnt.loops.prompts.call_llm",
        return_value=make_llm_result(stdout=_COMMENT_CLASSIFY_OK),
    ):
        resp, trace = prompts.run_classify_comment("p", engine="claude", model="m")
    assert resp.intent == "question"
    assert trace.error is None


def test_run_reply_comment_parses_llm_result():
    with patch(
        "mltgnt.loops.prompts.call_llm",
        return_value=make_llm_result(stdout=_COMMENT_REPLY_OK),
    ):
        resp, _trace = prompts.run_reply_comment("p", engine="claude", model="m")
    assert "進んで" in resp.reply


_ACTION_SCHEMAS = {
    "create_issue": {
        "type": "object",
        "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
        "required": ["title"],
        "additionalProperties": False,
    }
}


def test_validate_action_subtask_ok():
    resp = prompts.validate_decompose_payload(
        {
            "subtasks": [
                {
                    "id": "a1",
                    "title": "Create",
                    "kind": "action",
                    "action": {"name": "create_issue", "args": {"title": "x"}},
                    "depends": [],
                }
            ],
            "reasoning": "",
            "uncertain_flag": False,
        },
        max_subtasks=5,
        action_schemas=_ACTION_SCHEMAS,
    )
    assert resp.subtasks[0].kind == "action"
    assert resp.subtasks[0].action == {"name": "create_issue", "args": {"title": "x"}}
    assert resp.subtasks[0].prompt == ""


def test_validate_action_rejects_prompt_and_condition():
    with pytest.raises(ValueError, match="prompt"):
        prompts.validate_decompose_payload(
            {
                "subtasks": [
                    {
                        "id": "a1",
                        "title": "t",
                        "kind": "action",
                        "prompt": "nope",
                        "action": {"name": "create_issue", "args": {"title": "x"}},
                    }
                ],
                "reasoning": "",
                "uncertain_flag": False,
            },
            max_subtasks=5,
            action_schemas=_ACTION_SCHEMAS,
        )
    with pytest.raises(ValueError, match="condition"):
        prompts.validate_decompose_payload(
            {
                "subtasks": [
                    {
                        "id": "a1",
                        "title": "t",
                        "kind": "action",
                        "condition": {"type": "path_exists", "path": "x"},
                        "action": {"name": "create_issue", "args": {"title": "x"}},
                    }
                ],
                "reasoning": "",
                "uncertain_flag": False,
            },
            max_subtasks=5,
            action_schemas=_ACTION_SCHEMAS,
        )


def test_validate_action_unpublished_and_bad_args():
    with pytest.raises(ValueError, match="unpublished"):
        prompts.validate_decompose_payload(
            {
                "subtasks": [
                    {
                        "id": "a1",
                        "title": "t",
                        "kind": "action",
                        "action": {"name": "unknown", "args": {}},
                    }
                ],
                "reasoning": "",
                "uncertain_flag": False,
            },
            max_subtasks=5,
            action_schemas=_ACTION_SCHEMAS,
        )
    with pytest.raises(ValueError, match="missing required"):
        prompts.validate_decompose_payload(
            {
                "subtasks": [
                    {
                        "id": "a1",
                        "title": "t",
                        "kind": "action",
                        "action": {"name": "create_issue", "args": {}},
                    }
                ],
                "reasoning": "",
                "uncertain_flag": False,
            },
            max_subtasks=5,
            action_schemas=_ACTION_SCHEMAS,
        )


def test_build_decompose_includes_action_schemas():
    text = prompts.build_decompose_instruction(
        "body",
        iteration=1,
        max_subtasks=5,
        action_schemas=_ACTION_SCHEMAS,
    )
    assert "create_issue" in text
    assert "kind=action" in text
    assert '"action"' in text or "action.name" in text


def test_before_attempt_called_per_physical_try():
    calls: list[int] = []

    def before() -> None:
        calls.append(1)

    with patch(
        "mltgnt.loops.prompts.call_llm",
        side_effect=[
            make_llm_result(stdout="not json"),
            make_llm_result(stdout=_CLARIFY_OK),
        ],
    ):
        prompts.run_clarify(
            "prompt", engine="claude", model="m", before_attempt=before
        )
    assert len(calls) == 2
