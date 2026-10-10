"""mltgnt.agent — generic agent loop + decision-layer skeleton (#3318).

Design: Issue #287 / #3318
"""
from mltgnt.agent._runner import (
    REFLEXION_EXHAUSTED_TOOL,
    AgentResult,
    AgentRunner,
    LLMCaller,
    RetryConfig,
    ToolExecutor,
)
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
from mltgnt.agent.plan import Plan, PlanItem, build_plan_prompt, parse_plan
from mltgnt.agent.reflexion import DefaultReflexionEvaluator
from mltgnt.agent.work_loop import (
    FINISH_TOOL,
    GhdagSkillRunner,
    RepeatGuard,
    SkillRunner,
    TrackingCaller,
    WorkLoopConfig,
    WorkLoopDeadline,
    WorkLoopOutcome,
    is_plan_prompt,
    make_skill_tool,
    run_skill_contract,
    run_work_loop,
)

__all__ = [
    "AgentResult",
    "AgentRunner",
    "DefaultReflexionEvaluator",
    "DirectAgentResult",
    "FINISH_TOOL",
    "GhdagSkillRunner",
    "LLMCaller",
    "DispatchDecision",
    "MemoryWorkerResult",
    "MODE_DELEGATE",
    "MODE_REPLY",
    "Plan",
    "PlanItem",
    "PreflightContext",
    "REFLEXION_EXHAUSTED_TOOL",
    "RepeatGuard",
    "RetryConfig",
    "SkillWorkerResult",
    "SkillRunner",
    "ToolExecutor",
    "TrackingCaller",
    "WorkLoopConfig",
    "WorkLoopDeadline",
    "WorkLoopOutcome",
    "build_plan_prompt",
    "extract_artifact_references",
    "has_work_request",
    "is_create_request",
    "is_plan_prompt",
    "make_skill_tool",
    "make_dispatch_decision",
    "match_deferred_promise",
    "parse_plan",
    "run_preflight",
    "run_skill_contract",
    "run_work_loop",
    "should_force_delegate",
    "should_preempt_delegate",
]
