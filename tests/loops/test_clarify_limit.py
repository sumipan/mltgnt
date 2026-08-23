"""tests/loops/test_clarify_limit.py — clarify 上限到達時は decomposing へ進む。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from mltgnt.config import LoopsConfig
from mltgnt.interfaces.loops import HumanThreadRef
from mltgnt.loops.engine import LoopsEngine
from mltgnt.loops.models import LoopState
from mltgnt.loops import prompts
from mltgnt.loops import store
from tests.loops.fakes import FakeExecutor, FakeHumanChannel


def _config(tmp_path: Path) -> LoopsConfig:
    return LoopsConfig(
        objectives_dir=tmp_path / "objectives",
        state_dir=tmp_path / "state",
        status_dir=tmp_path / "status",
        jobs_dir=tmp_path / "jobs",
        exec_done_dir=tmp_path / "jobs" / "done",
        persona_dir=tmp_path / "personas",
        default_persona="mizuho",
        fallback_channel="C-fallback",
    )


def _engine(tmp_path, channel=None) -> LoopsEngine:
    return LoopsEngine(
        config=_config(tmp_path),
        human_channel=channel or FakeHumanChannel(),
        executor=FakeExecutor(),
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
    )


def _limit_state(*, clarification_context: list[str] | None = None) -> LoopState:
    """3 ラウンド回答済み（clarify_round == max）の clarifying 状態。"""
    ctx = clarification_context or [
        "Q: q1\nA: a1",
        "Q: q2\nA: a2",
        "Q: q3\nA: a3",
    ]
    return LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="clarifying",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=HumanThreadRef(channel_id="C1", thread_ts="123.456"),
        clarify_round=3,
        clarification_context=list(ctx),
        consecutive_errors=0,
        plan_approval=False,
        created_at="t",
        updated_at="t",
    )


@patch("mltgnt.loops.engine.load_persona")
def test_clarify_limit_transitions_to_decomposing(mock_persona, tmp_path):
    persona = MagicMock()
    persona.format_prompt.side_effect = lambda x, **_: x
    mock_persona.return_value = persona

    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel)
    state = _limit_state()
    store.save_state(_config(tmp_path).state_dir, state)

    engine.tick()

    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state is not None
    assert state.status == "decomposing"
    assert state.consecutive_errors == 0
    assert len(channel.notifies) == 1
    assert "仮定" in channel.notifies[0]["text"]

    events = store.read_events(_config(tmp_path).state_dir, "loop1")
    limit_events = [e for e in events if e["event"] == "clarify_limit"]
    assert len(limit_events) == 1
    assert limit_events[0]["data"]["rounds"] == 3


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_decompose")
def test_clarify_limit_notify_and_event_are_idempotent(mock_decompose, mock_persona, tmp_path):
    """decomposing 遷移後に tick を回しても notify / clarify_limit は増えない。"""
    persona = MagicMock()
    persona.format_prompt.side_effect = lambda x, **_: x
    mock_persona.return_value = persona
    mock_decompose.return_value = (
        prompts.DecomposeResponse(
            subtasks=[
                prompts.DecomposeSubtask(id="s1", title="T", kind="auto", prompt="p"),
            ],
            reasoning="",
            uncertain_flag=False,
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )

    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel)
    store.save_state(_config(tmp_path).state_dir, _limit_state())

    engine.tick()  # clarifying → decomposing (+ notify + event)
    engine.tick()  # decomposing → executing (no clarify_limit)

    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state is not None
    assert state.status == "executing"
    assert len(channel.notifies) == 1
    events = store.read_events(_config(tmp_path).state_dir, "loop1")
    assert len([e for e in events if e["event"] == "clarify_limit"]) == 1


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_decompose")
def test_clarify_limit_preserves_clarification_context_for_decompose(
    mock_decompose, mock_persona, tmp_path
):
    persona = MagicMock()
    persona.format_prompt.side_effect = lambda x, **_: x
    mock_persona.return_value = persona
    mock_decompose.return_value = (
        prompts.DecomposeResponse(
            subtasks=[],
            reasoning="",
            uncertain_flag=False,
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )

    ctx = ["Q: q1\nA: a1", "Q: q2\nA: a2", "Q: q3\nA: a3"]
    engine = _engine(tmp_path)
    store.save_state(_config(tmp_path).state_dir, _limit_state(clarification_context=ctx))

    engine.tick()
    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state is not None
    assert state.status == "decomposing"
    assert state.clarification_context == ctx

    engine.tick()
    assert mock_decompose.called
    # build_decompose_instruction 経由で context がプロンプトに入る
    prompt_arg = mock_decompose.call_args.args[0] if mock_decompose.call_args.args else (
        mock_decompose.call_args.kwargs.get("prompt")
        or mock_decompose.call_args[0][0]
    )
    # run_decompose の第1引数は persona 整形後プロンプト
    assert "q1" in prompt_arg and "a3" in prompt_arg
    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state is not None
    assert state.clarification_context == ctx


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_clarify")
def test_clarify_under_limit_still_asks(mock_clarify, mock_persona, tmp_path):
    """上限未満では従来どおり質問する（回帰）。"""
    persona = MagicMock()
    persona.format_prompt.side_effect = lambda x, **_: x
    mock_persona.return_value = persona
    mock_clarify.return_value = (
        prompts.ClarifyResponse(
            clear=False, question="Q?", reason="", reasoning="", uncertain_flag=False
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )

    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel)
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="clarifying",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        clarify_round=0,
        created_at="t",
        updated_at="t",
    )
    store.save_state(_config(tmp_path).state_dir, state)

    engine.tick()

    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state is not None
    assert state.status == "awaiting_answer"
    assert len(channel.asks) == 1
    assert state.clarify_round == 1
    events = store.read_events(_config(tmp_path).state_dir, "loop1")
    assert not any(e["event"] == "clarify_limit" for e in events)


@patch("mltgnt.loops.engine.load_persona")
def test_clarify_limit_does_not_increment_consecutive_errors_on_event_failure(
    mock_persona, tmp_path
):
    """clarify_limit のイベント記録失敗でも consecutive_errors は増えない。"""
    persona = MagicMock()
    persona.format_prompt.side_effect = lambda x, **_: x
    mock_persona.return_value = persona

    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel)
    state = _limit_state()
    store.save_state(_config(tmp_path).state_dir, state)

    original_append = store.append_event

    def _append(state_dir, loop_id, event, data, *, iteration):
        if event == "clarify_limit":
            raise OSError("disk full")
        return original_append(state_dir, loop_id, event, data, iteration=iteration)

    with patch("mltgnt.loops.engine.store.append_event", side_effect=_append):
        engine.tick()

    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state is not None
    assert state.status == "decomposing"
    assert state.consecutive_errors == 0
