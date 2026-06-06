from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from mltgnt.improvement.analyzer import FailurePattern, analyze_failures
from mltgnt.improvement.patch import PatchResult, apply_proposal
from mltgnt.improvement.proposal import ImprovementProposal, generate_proposals
from mltgnt.improvement.rollback import RollbackDecision


@dataclass
class CycleResult:
    patterns: list[FailurePattern]
    proposals: list[ImprovementProposal]
    period_start: date
    period_end: date
    patch_results: list[PatchResult] | None = None
    rollback_decision: RollbackDecision | None = None


def run_improvement_cycle(
    audit_path: Path,
    persona_dir: Path,
    skills_dir: Path,
    *,
    since_days: int = 7,
    today: date | None = None,
    eval_rollback: bool = False,
    repo_root: Path | None = None,
) -> CycleResult:
    if eval_rollback and repo_root is None:
        raise ValueError("repo_root is required when eval_rollback=True")

    if today is not None:
        period_end = today
    else:
        _as_of = os.environ.get("MLTGNT_AS_OF_DATE")
        period_end = date.fromisoformat(_as_of) if _as_of else date.today()
    period_start = period_end - timedelta(days=since_days)
    patterns = analyze_failures(audit_path, since=period_start, until=period_end)
    proposals = generate_proposals(patterns, persona_dir, skills_dir)

    if not eval_rollback:
        return CycleResult(
            patterns=patterns,
            proposals=proposals,
            period_start=period_start,
            period_end=period_end,
        )

    patch_results = [apply_proposal(proposal, repo_root) for proposal in proposals]
    return CycleResult(
        patterns=patterns,
        proposals=proposals,
        period_start=period_start,
        period_end=period_end,
        patch_results=patch_results,
    )
