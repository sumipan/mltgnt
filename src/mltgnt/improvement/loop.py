from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from mltgnt.improvement.analyzer import FailurePattern, analyze_failures
from mltgnt.improvement.proposal import ImprovementProposal, generate_proposals


@dataclass
class CycleResult:
    patterns: list[FailurePattern]
    proposals: list[ImprovementProposal]
    period_start: date
    period_end: date


def run_improvement_cycle(
    audit_path: Path,
    persona_dir: Path,
    skills_dir: Path,
    *,
    since_days: int = 7,
) -> CycleResult:
    period_end = date.today()
    period_start = period_end - timedelta(days=since_days)
    patterns = analyze_failures(audit_path, since=period_start, until=period_end)
    proposals = generate_proposals(patterns, persona_dir, skills_dir)
    return CycleResult(
        patterns=patterns,
        proposals=proposals,
        period_start=period_start,
        period_end=period_end,
    )
