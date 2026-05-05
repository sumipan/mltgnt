"""tests/tools/mltgnt_fugu/test_dag_bridge.py — DagBridge + Strategy基底ヘルパ テスト (AC-A1, AC-A3)"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.mltgnt_fugu.dag_bridge import (
    DagJobResult,
    DagJobSpec,
    FakeDagBridge,
    GhdagDagBridge,
)
from tools.mltgnt_fugu.mltgnt_strategy.base import Strategy
from tools.mltgnt_fugu.types import PlanState, StrategyInput, StrategyResult


# ---------------------------------------------------------------------------
# AC-A1: FakeDagBridge 単体テスト
# ---------------------------------------------------------------------------


def test_fake_bridge_submit_returns_uid() -> None:
    bridge = FakeDagBridge({"test_persona": "hello"})
    spec = DagJobSpec(prompt="x", persona_name="test_persona", engine="claude", model="sonnet")
    uid = bridge.submit(spec)
    assert uid == "fake-1"
    assert isinstance(uid, str)


def test_fake_bridge_wait_ok() -> None:
    bridge = FakeDagBridge({"test_persona": "hello"})
    spec = DagJobSpec(prompt="x", persona_name="test_persona", engine="claude", model="sonnet")
    uid = bridge.submit(spec)
    result = bridge.wait(uid)
    assert result == DagJobResult(uid="fake-1", status="ok", body="hello")


def test_fake_bridge_submit_and_wait_matches_submit_then_wait() -> None:
    responses = {"p": "world"}
    bridge1 = FakeDagBridge(responses)
    bridge2 = FakeDagBridge(responses)
    spec = DagJobSpec(prompt="x", persona_name="p", engine="claude", model="sonnet")

    uid = bridge1.submit(spec)
    expected = bridge1.wait(uid)

    result = bridge2.submit_and_wait(spec)
    assert result.status == expected.status
    assert result.body == expected.body


def test_fake_bridge_unregistered_persona() -> None:
    bridge = FakeDagBridge({})
    spec = DagJobSpec(prompt="x", persona_name="unknown", engine="claude", model="sonnet")
    result = bridge.submit_and_wait(spec)
    assert result.status == "error"
    assert "no response registered for persona: unknown" in result.body


def test_fake_bridge_unknown_uid() -> None:
    bridge = FakeDagBridge({"p": "ok"})
    result = bridge.wait("nonexistent")
    assert result.status == "error"
    assert result.body == "unknown uid"


def test_ghdag_bridge_instantiates() -> None:
    b = GhdagDagBridge(Path("/tmp/test.md"))
    spec = DagJobSpec(prompt="x", persona_name="p", engine="claude", model="sonnet")
    with pytest.raises(NotImplementedError):
        b.submit(spec)
    with pytest.raises(NotImplementedError):
        b.wait("uid-1")
    with pytest.raises(NotImplementedError):
        b.submit_and_wait(spec)


def test_fake_bridge_uid_increments() -> None:
    bridge = FakeDagBridge({"p": "r"})
    spec = DagJobSpec(prompt="x", persona_name="p", engine="claude", model="sonnet")
    uid1 = bridge.submit(spec)
    uid2 = bridge.submit(spec)
    assert uid1 == "fake-1"
    assert uid2 == "fake-2"


# ---------------------------------------------------------------------------
# AC-A3: Strategy 基底ヘルパ テスト
# ---------------------------------------------------------------------------


class _ConcreteStrategy(Strategy):
    """テスト用の最小具体化"""

    def execute(self, inp: StrategyInput) -> StrategyResult:
        return StrategyResult(
            response="",
            strategy_name="concrete",
            stop_reason="done",
            cost_usd=None,
            iterations=0,
        )


def _make_inp(metadata: dict, responses: dict | None = None) -> tuple[StrategyInput, FakeDagBridge]:
    bridge = FakeDagBridge(responses or {"persona": "resp"})
    inp = StrategyInput(
        prompt="test prompt",
        persona_name="persona",
        state=PlanState(metadata=metadata),
        dag=bridge,
    )
    return inp, bridge


def test_persona_call_uses_metadata_engine_model_timeout() -> None:
    inp, bridge = _make_inp({"engine": "openai", "model": "gpt-4", "timeout_s": 60})
    strategy = _ConcreteStrategy()
    strategy._persona_call(inp, "test prompt")
    submitted_spec = list(bridge._submitted.values())[0]
    assert submitted_spec.engine == "openai"
    assert submitted_spec.model == "gpt-4"
    assert submitted_spec.timeout_s == 60


def test_persona_call_uses_defaults_when_metadata_empty() -> None:
    inp, bridge = _make_inp({})
    strategy = _ConcreteStrategy()
    strategy._persona_call(inp, "test prompt")
    submitted_spec = list(bridge._submitted.values())[0]
    assert submitted_spec.engine == "claude"
    assert submitted_spec.model == "sonnet"
    assert submitted_spec.timeout_s == 120


def test_judge_call_uses_fixed_haiku_settings() -> None:
    inp, bridge = _make_inp({"engine": "openai", "model": "gpt-4", "timeout_s": 999})
    strategy = _ConcreteStrategy()
    strategy._judge_call(inp, "judge prompt")
    submitted_spec = list(bridge._submitted.values())[0]
    assert submitted_spec.engine == "claude"
    assert submitted_spec.model == "haiku"
    assert submitted_spec.timeout_s == 30


def test_persona_call_passes_depends() -> None:
    inp, bridge = _make_inp({})
    strategy = _ConcreteStrategy()
    strategy._persona_call(inp, "p", depends=("uid-1",))
    submitted_spec = list(bridge._submitted.values())[0]
    assert submitted_spec.depends == ("uid-1",)
