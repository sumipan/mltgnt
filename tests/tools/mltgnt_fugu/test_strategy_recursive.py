"""tests/tools/mltgnt_fugu/test_strategy_recursive.py — RecursiveStrategy fixture (AC-A4 Phase A)"""
from __future__ import annotations

import pytest

from tools.mltgnt_fugu.dag_bridge import FakeDagBridge
from tools.mltgnt_fugu.types import PlanState, StrategyInput


@pytest.fixture
def recursive_inp() -> StrategyInput:
    return StrategyInput(
        prompt="test prompt",
        persona_name="recursive_persona",
        state=PlanState(),
        dag=FakeDagBridge({}),
    )


def test_recursive_inp_fixture_has_dag(recursive_inp: StrategyInput) -> None:
    assert isinstance(recursive_inp.dag, FakeDagBridge)
