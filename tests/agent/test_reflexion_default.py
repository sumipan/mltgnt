"""tests/agent/test_reflexion_default.py -- DefaultReflexionEvaluator (#3855)."""
from __future__ import annotations

from mltgnt.agent.reflexion import DefaultReflexionEvaluator


def _entry(tool: str, args: dict, result: str = "ok") -> dict:
    return {"tool": tool, "args": args, "result": result}


def _call(ev, tool, args, result, trace):
    return ev("prompt", tool, args, result, trace)


def test_error_result_retries_with_body():
    ev = DefaultReflexionEvaluator()
    trace = [_entry("t", {"x": 1}, "[ERROR] boom")]
    verdict = _call(ev, "t", {"x": 1}, "[ERROR] boom", trace)
    assert verdict.should_retry is True
    assert "boom" in verdict.feedback


def test_error_result_with_leading_whitespace():
    ev = DefaultReflexionEvaluator()
    verdict = _call(ev, "t", {}, "  \n[ERROR] bad", [_entry("t", {})])
    assert verdict.should_retry is True


def test_error_excerpt_is_limited():
    ev = DefaultReflexionEvaluator()
    body = "[ERROR] " + "x" * 2000
    verdict = _call(ev, "t", {}, body, [_entry("t", {})])
    assert verdict.should_retry is True
    assert "x" * 600 not in verdict.feedback


def test_failure_marker():
    ev = DefaultReflexionEvaluator(failure_markers=("FATAL",))
    verdict = _call(ev, "t", {}, "log line FATAL here", [_entry("t", {})])
    assert verdict.should_retry is True
    assert "FATAL" in verdict.feedback


def test_failure_marker_default_is_empty():
    ev = DefaultReflexionEvaluator()
    verdict = _call(ev, "t", {}, "log line FATAL here", [_entry("t", {})])
    assert verdict.should_retry is False


def test_repeat_within_window_retries():
    ev = DefaultReflexionEvaluator()
    trace = [
        _entry("t", {"a": 1, "b": 2}),
        _entry("u", {}),
        _entry("v", {}),
        _entry("t", {"b": 2, "a": 1}),  # current call
    ]
    verdict = _call(ev, "t", {"b": 2, "a": 1}, "ok", trace)
    assert verdict.should_retry is True
    assert verdict.feedback


def test_repeat_outside_window_does_not_retry():
    ev = DefaultReflexionEvaluator()
    trace = [
        _entry("t", {"a": 1}),
        _entry("u", {}),
        _entry("v", {}),
        _entry("w", {}),
        _entry("t", {"a": 1}),  # current call
    ]
    verdict = _call(ev, "t", {"a": 1}, "ok", trace)
    assert verdict.should_retry is False


def test_current_entry_alone_is_not_a_repeat():
    ev = DefaultReflexionEvaluator()
    verdict = _call(ev, "t", {"a": 1}, "ok", [_entry("t", {"a": 1})])
    assert verdict.should_retry is False


def test_clean_result_no_retry():
    ev = DefaultReflexionEvaluator(failure_markers=("FATAL",))
    trace = [_entry("u", {}), _entry("t", {"a": 1})]
    verdict = _call(ev, "t", {"a": 1}, "all good", trace)
    assert verdict.should_retry is False
    assert verdict.feedback == ""
