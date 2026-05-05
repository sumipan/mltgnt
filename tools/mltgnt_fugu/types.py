from __future__ import annotations

from dataclasses import dataclass, field

from .dag_bridge import DagBridge


@dataclass
class PlanState:
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class StrategyInput:
    prompt: str
    persona_name: str
    state: PlanState
    dag: DagBridge
    context: dict[str, str] = field(default_factory=dict)


@dataclass
class StrategyResult:
    response: str
    strategy_name: str
    stop_reason: str
    cost_usd: float | None
    iterations: int
    trace_uids: list[str] = field(default_factory=list)
    final_uid: str | None = None
