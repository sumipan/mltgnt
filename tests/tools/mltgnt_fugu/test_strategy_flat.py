"""tests/tools/mltgnt_fugu/test_strategy_flat.py — FlatStrategy fixture (AC-A4 Phase A)"""
from __future__ import annotations

import pytest

from tools.mltgnt_fugu.dag_bridge import FakeDagBridge
from tools.mltgnt_fugu.types import PlanState, StrategyInput


@pytest.fixture
def flat_inp() -> StrategyInput:
    return StrategyInput(
        prompt="test prompt",
        persona_name="flat_persona",
        state=PlanState(),
        dag=FakeDagBridge({}),
    )


def test_flat_inp_fixture_has_dag(flat_inp: StrategyInput) -> None:
    assert isinstance(flat_inp.dag, FakeDagBridge)
