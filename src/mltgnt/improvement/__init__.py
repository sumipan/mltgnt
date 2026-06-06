"""mltgnt.improvement public API."""

from mltgnt.improvement.analyzer import FailurePattern, analyze_failures
from mltgnt.improvement.loop import CycleResult, run_improvement_cycle
from mltgnt.improvement.patch import PatchResult
from mltgnt.improvement.proposal import ImprovementProposal, generate_proposals
from mltgnt.improvement.rollback import (
    RollbackDecision,
    evaluate_cycle_outcome,
    evaluate_rollback,
    execute_rollback,
)

__all__ = [
    "FailurePattern",
    "analyze_failures",
    "ImprovementProposal",
    "generate_proposals",
    "PatchResult",
    "RollbackDecision",
    "evaluate_cycle_outcome",
    "evaluate_rollback",
    "execute_rollback",
    "CycleResult",
    "run_improvement_cycle",
]
