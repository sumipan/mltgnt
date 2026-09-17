"""mltgnt.agent — generic agent loop + decision-layer skeleton (#3318).

Design: Issue #287 / #3318
"""
from mltgnt.agent._runner import AgentResult, AgentRunner
from mltgnt.agent.deterministic_gate import (
    extract_artifact_references,
    has_work_request,
    is_create_request,
    match_deferred_promise,
    should_force_delegate,
    should_preempt_delegate,
)
from mltgnt.agent.dispatch_decision import (
    MODE_DELEGATE,
    MODE_REPLY,
    DirectAgentResult,
    DispatchDecision,
    make_dispatch_decision,
)
from mltgnt.agent.dispatch_preflight import (
    MemoryWorkerResult,
    PreflightContext,
    SkillWorkerResult,
    run_preflight,
)

__all__ = [
    "AgentResult",
    "AgentRunner",
    "DirectAgentResult",
    "DispatchDecision",
    "MemoryWorkerResult",
    "MODE_DELEGATE",
    "MODE_REPLY",
    "PreflightContext",
    "SkillWorkerResult",
    "extract_artifact_references",
    "has_work_request",
    "is_create_request",
    "make_dispatch_decision",
    "match_deferred_promise",
    "run_preflight",
    "should_force_delegate",
    "should_preempt_delegate",
]
