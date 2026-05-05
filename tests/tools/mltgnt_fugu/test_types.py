"""tests/tools/mltgnt_fugu/test_types.py — StrategyInput / StrategyResult テスト (AC-A2)"""
from __future__ import annotations

import pytest

from tools.mltgnt_fugu.dag_bridge import FakeDagBridge
from tools.mltgnt_fugu.types import PlanState, StrategyInput, StrategyResult


# ---------------------------------------------------------------------------
# AC-A2: StrategyInput
# ---------------------------------------------------------------------------


def test_strategy_input_requires_dag() -> None:
    with pytest.raises(TypeError):
        StrategyInput(prompt="x", persona_name="p", state=PlanState())  # type: ignore[call-arg]


def test_strategy_input_with_dag() -> None:
    inp = StrategyInput(
        prompt="x",
        persona_name="p",
        state=PlanState(),
        dag=FakeDagBridge({}),
    )
    assert inp.prompt == "x"
    assert inp.persona_name == "p"
    assert isinstance(inp.state, PlanState)
    assert isinstance(inp.dag, FakeDagBridge)
    assert inp.context == {}


def test_strategy_input_with_context() -> None:
    inp = StrategyInput(
        prompt="x",
        persona_name="p",
        state=PlanState(),
        dag=FakeDagBridge({}),
        context={"key": "value"},
    )
    assert inp.context == {"key": "value"}


# ---------------------------------------------------------------------------
# AC-A2: StrategyResult
# ---------------------------------------------------------------------------


def test_strategy_result_cost_usd_none() -> None:
    result = StrategyResult(
        response="x",
        strategy_name="flat",
        stop_reason="done",
        cost_usd=None,
        iterations=1,
    )
    assert result.cost_usd is None
    assert result.trace_uids == []
    assert result.final_uid is None


def test_strategy_result_with_trace_fields() -> None:
    result = StrategyResult(
        response="x",
        strategy_name="flat",
        stop_reason="done",
        cost_usd=0.5,
        iterations=2,
        trace_uids=["uid-1", "uid-2"],
        final_uid="uid-2",
    )
    assert result.cost_usd == 0.5
    assert result.trace_uids == ["uid-1", "uid-2"]
    assert result.final_uid == "uid-2"


def test_plan_state_default_metadata() -> None:
    state = PlanState()
    assert state.metadata == {}


def test_plan_state_with_metadata() -> None:
    state = PlanState(metadata={"engine": "claude", "model": "sonnet"})
    assert state.metadata["engine"] == "claude"
