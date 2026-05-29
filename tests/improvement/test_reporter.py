from __future__ import annotations

from datetime import date

from mltgnt.improvement import FailurePattern, ImprovementProposal
from mltgnt.improvement.loop import CycleResult
from mltgnt.improvement.reporter import format_cycle_report


def _pattern(pattern_id: str, category: str, count: int) -> FailurePattern:
    return FailurePattern(
        pattern_id=pattern_id,
        category=category,
        count=count,
        example_correlation_ids=[f"{pattern_id}-corr"],
        affected_persona="タチコマ",
        affected_skill=None,
    )


def _proposal(proposal_id: str) -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id=proposal_id,
        target_type="persona",
        target_name="タチコマ",
        action="adjust_section",
        description="テスト提案",
        diff_preview="+ hint",
        confidence=0.8,
        source_patterns=["p1"],
    )


def test_format_cycle_report_with_patterns_and_proposals() -> None:
    result = CycleResult(
        patterns=[_pattern("p1", "triage_error", 3), _pattern("p2", "timeout", 4)],
        proposals=[_proposal("proposal:p1")],
        period_start=date(2026, 5, 22),
        period_end=date(2026, 5, 29),
    )

    report = format_cycle_report(result)

    assert "# サマリ" in report
    assert "# 失敗パターン一覧" in report
    assert "# 改善提案一覧" in report
    assert "2026-05-22" in report
    assert "2026-05-29" in report
    assert "category" in report
    assert "count" in report
    assert "example_correlation_ids" in report
    assert "target_type" in report
    assert "target_name" in report
    assert "action" in report
    assert "description" in report
    assert "confidence" in report
    assert "triage_error" in report
    assert "adjust_section" in report


def test_format_cycle_report_empty_result() -> None:
    result = CycleResult(
        patterns=[],
        proposals=[],
        period_start=date(2026, 5, 22),
        period_end=date(2026, 5, 29),
    )

    report = format_cycle_report(result)

    assert "対象期間に失敗パターンは検出されませんでした" in report
    assert "| category |" not in report
    assert "| target_type |" not in report
    assert "2026-05-22" in report
    assert "2026-05-29" in report
