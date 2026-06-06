"""
tests/test_skill/test_agentic_discover.py — AgenticSkillDiscoverer のユニットテスト。

設計: Issue #1922
"""
from __future__ import annotations

from pathlib import Path

from mltgnt.routing.agentic_discover import AgenticSkillDiscoverer
from mltgnt.skill.models import SkillMeta


def _meta(
    name: str,
    description: str,
    *,
    triggers: list[str] | None = None,
) -> SkillMeta:
    return SkillMeta(
        name=name,
        description=description,
        argument_hint="",
        model=None,
        path=Path(f"/fake/skills/{name}/SKILL.md"),
        triggers=triggers or [],
    )


def _catalog() -> dict[str, SkillMeta]:
    return {
        "calendar": _meta(
            "calendar",
            "カレンダーの予定を確認・追加する",
            triggers=["予定", "スケジュール", "カレンダー"],
        ),
        "diary-draft": _meta(
            "diary-draft",
            "日記の下書きを作成する",
            triggers=["日記", "下書き"],
        ),
        "review": _meta(
            "review",
            "週次レビューを実行する",
            triggers=["振り返り", "レビュー"],
        ),
    }


def _make_llm_responses(*responses: str):
    it = iter(responses)

    def llm_call(_prompt: str) -> str:
        return next(it)

    return llm_call


def test_discover_selected():
    discoverer = AgenticSkillDiscoverer(
        _make_llm_responses("SELECTED\ncalendar"),
        max_iterations=3,
    )

    result = discoverer.discover(
        user_input="予定を確認して",
        skill_catalog=_catalog(),
    )

    assert result.kind == "selected"
    assert result.skill is not None
    assert result.skill.name == "calendar"
    assert len(result.trace) == 1
    assert result.trace[0].verdict == "SELECTED"


def test_discover_empty_catalog():
    discoverer = AgenticSkillDiscoverer(lambda _: "SELECTED\ncalendar")

    result = discoverer.discover(
        user_input="予定を確認して",
        skill_catalog={},
    )

    assert result.kind == "unresolved"
    assert result.skill is None
    assert result.candidates == []
    assert result.trace == []


def test_discover_persona_skills_empty_after_filter():
    discoverer = AgenticSkillDiscoverer(lambda _: "SELECTED\ncalendar")

    result = discoverer.discover(
        user_input="予定を確認して",
        skill_catalog=_catalog(),
        persona_skills=["nonexistent"],
    )

    assert result.kind == "unresolved"
    assert result.trace == []


def test_discover_max_iterations_ambiguous():
    discoverer = AgenticSkillDiscoverer(
        _make_llm_responses(
            "NEED_MORE\n予定 確認",
            "NEED_MORE\nスケジュール",
            "NEED_MORE\nカレンダー",
        ),
        max_iterations=3,
    )

    result = discoverer.discover(
        user_input="予定を確認して",
        skill_catalog=_catalog(),
    )

    assert result.kind == "ambiguous"
    assert result.skill is None
    assert len(result.candidates) > 0
    assert len(result.trace) == 3
    assert all(r.verdict == "NEED_MORE" for r in result.trace)


def test_discover_unresolved_verdict():
    discoverer = AgenticSkillDiscoverer(
        _make_llm_responses("UNRESOLVED"),
        max_iterations=3,
    )

    result = discoverer.discover(
        user_input="予定を確認して",
        skill_catalog=_catalog(),
    )

    assert result.kind == "unresolved"
    assert result.skill is None
    assert len(result.candidates) > 0
    assert result.trace[0].verdict == "UNRESOLVED"


def test_discover_persona_skills_filter():
    discoverer = AgenticSkillDiscoverer(
        _make_llm_responses("SELECTED\ncalendar"),
        max_iterations=3,
    )

    result = discoverer.discover(
        user_input="予定を確認して",
        skill_catalog=_catalog(),
        persona_skills=["calendar"],
    )

    assert result.kind == "selected"
    assert result.skill is not None
    assert result.skill.name == "calendar"
