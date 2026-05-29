"""mltgnt.improvement public API."""

from mltgnt.improvement.analyzer import FailurePattern, analyze_failures
from mltgnt.improvement.loop import CycleResult, run_improvement_cycle
from mltgnt.improvement.proposal import ImprovementProposal, generate_proposals

__all__ = [
    "FailurePattern",
    "analyze_failures",
    "ImprovementProposal",
    "generate_proposals",
    "CycleResult",
    "run_improvement_cycle",
]
