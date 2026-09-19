"""
tests/test_sufficiency.py — _sufficiency.py  unit tests

TC1: SUFFICIENT Response
TC2: INSUFFICIENT + MEMORY Response
TC3: INSUFFICIENT +ILLILL Response
TC4: INSUFFICIENT + too few lines (fail-safe)
TC5: empty response (fail-safe)
TC6: unknown source (fail-safe)
TC7: unknown format (fail-safe)
TC8: LLM exception propagation
TC9: rewritten_query compat property(INSUFFICIENT/MEMORY)
TC10: rewritten_query compat property(SUFFICIENT)
"""
from __future__ import annotations

import logging
import pytest
from mltgnt.memory._sufficiency import (
    judge_for_discover,
    judge_sufficiency,
)


def _llm(response: str):
    """llm_call factory that returns a fixed response"""
    def call(_prompt: str) -> str:
        return response
    return call


# ---------------------------------------------------------------------------
# TC1: SUFFICIENT
# ---------------------------------------------------------------------------


def test_tc1_sufficient():
    result = judge_sufficiency("question", "info", _llm("SUFFICIENT"))
    assert result.sufficient is True
    assert result.action is None


# ---------------------------------------------------------------------------
# TC2: INSUFFICIENT + MEMORY
# ---------------------------------------------------------------------------


def test_tc2_insufficient_memory():
    response = "INSUFFICIENT\nMEMORY\nproject progress"
    result = judge_sufficiency("question", "info", _llm(response))
    assert result.sufficient is False
    assert result.action is not None
    assert result.action.source == "memory"
    assert result.action.query == "project progress"


# ---------------------------------------------------------------------------
# TC3: INSUFFICIENT + SKILL
# ---------------------------------------------------------------------------


def test_tc3_insufficient_skill():
    response = "INSUFFICIENT\nSKILL\ndeploy steps"
    result = judge_sufficiency("question", "info", _llm(response))
    assert result.sufficient is False
    assert result.action is not None
    assert result.action.source == "skill"
    assert result.action.query == "deploy steps"


# ---------------------------------------------------------------------------
# TC4: INSUFFICIENT + too few lines (fail-safe)
# ---------------------------------------------------------------------------


def test_tc4_insufficient_missing_lines(caplog):
    with caplog.at_level(logging.WARNING, logger="mltgnt.memory._sufficiency"):
        result = judge_sufficiency("question", "info", _llm("INSUFFICIENT\nMEMORY"))
    assert result.sufficient is True
    assert result.action is None
    assert any("missing" in r.message or "SUFFICIENT" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# TC5: empty response (fail-safe)
# ---------------------------------------------------------------------------


def test_tc5_empty_response(caplog):
    with caplog.at_level(logging.WARNING, logger="mltgnt.memory._sufficiency"):
        result = judge_sufficiency("question", "info", _llm(""))
    assert result.sufficient is True
    assert result.action is None


# ---------------------------------------------------------------------------
# TC6: unknown source (fail-safe)
# ---------------------------------------------------------------------------


def test_tc6_unknown_source(caplog):
    response = "INSUFFICIENT\nWEB\nsearch query"
    with caplog.at_level(logging.WARNING, logger="mltgnt.memory._sufficiency"):
        result = judge_sufficiency("question", "info", _llm(response))
    assert result.sufficient is True
    assert result.action is None


# ---------------------------------------------------------------------------
# TC7: unknown format (fail-safe)
# ---------------------------------------------------------------------------


def test_tc7_unexpected_format(caplog):
    with caplog.at_level(logging.WARNING, logger="mltgnt.memory._sufficiency"):
        result = judge_sufficiency("question", "info", _llm("MAYBE"))
    assert result.sufficient is True
    assert result.action is None


# ---------------------------------------------------------------------------
# TC8: LLM exception propagation
# ---------------------------------------------------------------------------


def test_tc8_llm_exception():
    def failing_llm(_prompt: str) -> str:
        raise RuntimeError("API error")

    with pytest.raises(RuntimeError, match="API error"):
        judge_sufficiency("question", "info", failing_llm)


# ---------------------------------------------------------------------------
# TC9: rewritten_query compat property(INSUFFICIENT/MEMORY)
# ---------------------------------------------------------------------------


def test_tc9_rewritten_query_insufficient():
    response = "INSUFFICIENT\nMEMORY\nlast week progress"
    result = judge_sufficiency("question", "info", _llm(response))
    assert result.rewritten_query == "last week progress"


# ---------------------------------------------------------------------------
# TC10: rewritten_query compat property(SUFFICIENT)
# ---------------------------------------------------------------------------


def test_tc10_rewritten_query_sufficient():
    result = judge_sufficiency("question", "info", _llm("SUFFICIENT"))
    assert result.rewritten_query is None


# ---------------------------------------------------------------------------
# judge_for_discover
# ---------------------------------------------------------------------------


def test_judge_for_discover_selected():
    response = "SELECTED\ncalendar"
    result = judge_for_discover(
        "calendar check",
        "calendar: schedule check (score: 0.85)",
        ["calendar", "diary-draft", "review"],
        _llm(response),
    )
    assert result.kind == "selected"
    assert result.skill_name == "calendar"
    assert result.next_query is None


def test_judge_for_discover_need_more():
    response = "NEED_MORE\nschedule schedule"
    result = judge_for_discover(
        "schedule",
        "calendar: schedule check (score: 0.50)",
        ["calendar", "diary-draft"],
        _llm(response),
    )
    assert result.kind == "need_more"
    assert result.next_query == "schedule schedule"
    assert result.skill_name is None


def test_judge_for_discover_unresolved():
    result = judge_for_discover(
        "question",
        "candidate info",
        ["calendar", "review"],
        _llm("UNRESOLVED"),
    )
    assert result.kind == "unresolved"
    assert result.reason == "no_match"


def test_judge_for_discover_parse_error(caplog):
    with caplog.at_level(logging.WARNING, logger="mltgnt.memory._sufficiency"):
        result = judge_for_discover(
            "question",
            "candidate info",
            ["calendar"],
            _llm("INVALID"),
        )
    assert result.kind == "unresolved"
    assert result.reason == "parse_error"
    assert result.top_candidates == []


def test_judge_for_discover_llm_exception():
    def failing_llm(_prompt: str) -> str:
        raise RuntimeError("API error")

    with pytest.raises(RuntimeError, match="API error"):
        judge_for_discover("question", "candidate", ["calendar"], failing_llm)
