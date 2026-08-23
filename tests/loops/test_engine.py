"""tests/loops/test_engine.py — 状態遷移テスト。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.config import LoopsConfig
from mltgnt.loops.engine import LoopsEngine
from mltgnt.loops.models import LoopState, PendingQuestion, Subtask
from mltgnt.loops.objective import Objective
from mltgnt.loops import store
from mltgnt.loops import prompts
from mltgnt.interfaces.loops import HumanThreadRef, StepPoll, StepSubmission
from tests.loops.fakes import FakeExecutor, FakeHumanChannel


def _thread() -> HumanThreadRef:
    return HumanThreadRef(channel_id="C1", thread_ts="123.456")


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


def _objective(loop_id: str = "loop1", *, plan_approval: bool = False) -> Objective:
    return Objective(
        loop_id=loop_id,
        title="Title",
        body="Do something",
        agent="mizuho",
        max_iterations=5,
        status="active",
        path=Path(f"/tmp/{loop_id}.md"),
        content_hash="hash1",
        plan_approval=plan_approval,
    )


def _engine(tmp_path, channel=None, executor=None) -> LoopsEngine:
    cfg = _config(tmp_path)
    return LoopsEngine(
        config=cfg,
        human_channel=channel or FakeHumanChannel(),
        executor=executor or FakeExecutor(),
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
    )


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_clarify")
def test_clarify_ask_then_answer(mock_clarify, mock_persona, tmp_path):
    persona = MagicMock()
    persona.format_prompt.side_effect = lambda x, **_: x
    mock_persona.return_value = persona

    mock_clarify.return_value = (
        prompts.ClarifyResponse(
            clear=False, question="公開日はいつですか", reason="", reasoning="", uncertain_flag=False
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )

    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel)
    engine.start_loop(_objective())

    engine.tick()
    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state.status == "awaiting_answer"
    assert len(channel.asks) == 1
    qid = channel.asks[0]["question_id"]

    inbox = store._inbox_dir(_config(tmp_path).state_dir, "loop1")
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "001-a.json").write_text(
        json.dumps(
            {
                "kind": "answer",
                "message_id": "m1",
                "question_id": qid,
                "text": "来月",
                "received_at": "2026-08-20T12:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )

    engine.tick()
    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state.status == "clarifying"
    assert state.clarification_context == ["Q: 公開日はいつですか\nA: 来月"]


@patch("mltgnt.loops.engine.load_persona")
def test_persona_missing_fails_with_fallback(mock_persona, tmp_path):
    mock_persona.side_effect = FileNotFoundError("missing")
    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel)
    engine.start_loop(_objective())
    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    state.status = "clarifying"
    store.save_state(_config(tmp_path).state_dir, state)

    engine.tick()
    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state.status == "failed"
    assert len(channel.fallbacks) == 1


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_decompose")
def test_decompose_rejects_too_many_subtasks(mock_decompose, mock_persona, tmp_path):
    persona = MagicMock()
    persona.format_prompt.side_effect = lambda x, **_: x
    mock_persona.return_value = persona
    mock_decompose.side_effect = ValueError("too many subtasks")

    engine = _engine(tmp_path)
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="decomposing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        created_at="t",
        updated_at="t",
    )
    store.save_state(_config(tmp_path).state_dir, state)

    engine.tick()
    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state.consecutive_errors == 1


def test_cancel_inbox(tmp_path):
    engine = _engine(tmp_path)
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="executing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        created_at="t",
        updated_at="t",
    )
    store.save_state(_config(tmp_path).state_dir, state)
    inbox = store._inbox_dir(_config(tmp_path).state_dir, "loop1")
    inbox.mkdir(parents=True)
    (inbox / "001-c.json").write_text(
        json.dumps(
            {
                "kind": "cancel",
                "message_id": "c1",
                "question_id": "",
                "text": "",
                "received_at": "2026-08-20T12:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )
    engine.tick()
    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state.status == "cancelled"


@patch("mltgnt.loops.engine.load_persona")
def test_consecutive_errors_reset(mock_persona, tmp_path):
    persona = MagicMock()
    persona.format_prompt.side_effect = lambda x, **_: x
    mock_persona.return_value = persona

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
        consecutive_errors=2,
        created_at="t",
        updated_at="t",
    )
    store.save_state(_config(tmp_path).state_dir, state)

    with patch("mltgnt.loops.engine.prompts.run_clarify") as mock_clarify:
        mock_clarify.return_value = (
            prompts.ClarifyResponse(
                clear=False, question="Q", reason="", reasoning="", uncertain_flag=False
            ),
            prompts.LlmTrace("", "", {}, "", {}, {}, False),
        )
        engine.tick()

    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state.consecutive_errors == 0


def test_poll_errors_fail_after_three_consecutive_ticks(tmp_path):
    executor = FakeExecutor()
    executor.poll = MagicMock(side_effect=RuntimeError("poll unavailable"))
    engine = _engine(tmp_path, executor=executor)
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="executing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        subtasks=[Subtask(
            id="s1",
            title="S1",
            kind="auto",
            prompt="do it",
            status="running",
            submission=StepSubmission(
                uuid="u1",
                result_filename="r1.md",
                submitted_at="2026-08-20T12:00:00+09:00",
                reused=False,
            ),
        )],
        current_subtask_id="s1",
    )
    store.save_state(_config(tmp_path).state_dir, state)

    engine.tick()
    engine.tick()
    engine.tick()

    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state.status == "failed"
    assert state.consecutive_errors == 3


def _persona_with_ops(*, engine: str = "", model: str = "") -> MagicMock:
    persona = MagicMock()
    persona.format_prompt.side_effect = lambda x, **_: x
    persona.fm.engine = engine
    persona.fm.model = model
    return persona


def _config_with_llm(tmp_path: Path, **overrides) -> LoopsConfig:
    kwargs = dict(
        objectives_dir=tmp_path / "objectives",
        state_dir=tmp_path / "state",
        status_dir=tmp_path / "status",
        jobs_dir=tmp_path / "jobs",
        exec_done_dir=tmp_path / "jobs" / "done",
        persona_dir=tmp_path / "personas",
        default_persona="mizuho",
        fallback_channel="C-fallback",
        llm_engine="cursor",
        llm_model="auto",
        subtask_engine="cursor",
        subtask_model="auto",
    )
    kwargs.update(overrides)
    return LoopsConfig(**kwargs)


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_clarify")
def test_clarify_uses_persona_engine_model(mock_clarify, mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops(engine="claude", model="claude-sonnet-4-6")
    mock_clarify.return_value = (
        prompts.ClarifyResponse(
            clear=True, question="", reason="", reasoning="", uncertain_flag=False
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    cfg = _config_with_llm(tmp_path)
    engine = LoopsEngine(
        config=cfg,
        human_channel=FakeHumanChannel(),
        executor=FakeExecutor(),
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
    )
    engine.start_loop(_objective())
    engine.tick()

    mock_clarify.assert_called_once()
    assert mock_clarify.call_args.kwargs["engine"] == "claude"
    assert mock_clarify.call_args.kwargs["model"] == "claude-sonnet-4-6"


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_decompose")
def test_decompose_uses_persona_engine_model(mock_decompose, mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops(engine="claude", model="claude-sonnet-4-6")
    mock_decompose.return_value = (
        prompts.DecomposeResponse(
            subtasks=[prompts.DecomposeSubtask(id="s1", title="T", kind="auto", prompt="p")],
            reasoning="",
            uncertain_flag=False,
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    cfg = _config_with_llm(tmp_path)
    engine = LoopsEngine(
        config=cfg,
        human_channel=FakeHumanChannel(),
        executor=FakeExecutor(),
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
    )
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="decomposing",
        iteration=1,
        max_iterations=5,
        persona="tachikoma",
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg.state_dir, state)
    engine.tick()

    assert mock_decompose.call_args.kwargs["engine"] == "claude"
    assert mock_decompose.call_args.kwargs["model"] == "claude-sonnet-4-6"


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_evaluate")
def test_evaluate_uses_persona_engine_model(mock_evaluate, mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops(engine="claude", model="claude-sonnet-4-6")
    mock_evaluate.return_value = (
        prompts.EvaluateResponse(
            achieved=True,
            score=100,
            summary="done",
            next_focus="",
            reasoning="",
            uncertain_flag=False,
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    cfg = _config_with_llm(tmp_path)
    engine = LoopsEngine(
        config=cfg,
        human_channel=FakeHumanChannel(),
        executor=FakeExecutor(),
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
    )
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="evaluating",
        iteration=1,
        max_iterations=5,
        persona="tachikoma",
        subtasks=[Subtask(id="s1", title="S1", kind="auto", prompt="p", status="success", result="ok")],
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg.state_dir, state)
    engine.tick()

    assert mock_evaluate.call_args.kwargs["engine"] == "claude"
    assert mock_evaluate.call_args.kwargs["model"] == "claude-sonnet-4-6"


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_clarify")
def test_clarify_falls_back_to_config_when_persona_ops_empty(mock_clarify, mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops(engine="", model="")
    mock_clarify.return_value = (
        prompts.ClarifyResponse(
            clear=True, question="", reason="", reasoning="", uncertain_flag=False
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    cfg = _config_with_llm(tmp_path)
    engine = LoopsEngine(
        config=cfg,
        human_channel=FakeHumanChannel(),
        executor=FakeExecutor(),
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
    )
    engine.start_loop(_objective())
    engine.tick()

    assert mock_clarify.call_args.kwargs["engine"] == "cursor"
    assert mock_clarify.call_args.kwargs["model"] == "auto"


@patch("mltgnt.loops.engine.load_persona")
def test_submit_uses_persona_engine_model(mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops(engine="claude", model="claude-sonnet-4-6")
    executor = FakeExecutor()
    cfg = _config_with_llm(tmp_path)
    engine = LoopsEngine(
        config=cfg,
        human_channel=FakeHumanChannel(),
        executor=executor,
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
    )
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="executing",
        iteration=1,
        max_iterations=5,
        persona="tachikoma",
        subtasks=[Subtask(id="s1", title="S1", kind="auto", prompt="do it", status="pending")],
        current_subtask_id="s1",
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg.state_dir, state)
    engine.tick()

    assert len(executor.submit_kwargs) == 1
    assert executor.submit_kwargs[0]["engine"] == "claude"
    assert executor.submit_kwargs[0]["model"] == "claude-sonnet-4-6"


@patch("mltgnt.loops.engine.load_persona")
def test_submit_falls_back_to_subtask_config_when_persona_ops_empty(mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops(engine="", model="")
    executor = FakeExecutor()
    cfg = _config_with_llm(tmp_path)
    engine = LoopsEngine(
        config=cfg,
        human_channel=FakeHumanChannel(),
        executor=executor,
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
    )
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="executing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        subtasks=[Subtask(id="s1", title="S1", kind="auto", prompt="do it", status="pending")],
        current_subtask_id="s1",
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg.state_dir, state)
    engine.tick()

    assert executor.submit_kwargs[0]["engine"] == "cursor"
    assert executor.submit_kwargs[0]["model"] == "auto"


def test_resolve_llm_falls_back_on_persona_load_failure_and_does_not_raise(tmp_path):
    """読込失敗時は例外を上げず config にフォールバックする（呼び出し側はループ継続可能）。"""
    cfg = _config_with_llm(tmp_path)
    engine = LoopsEngine(
        config=cfg,
        human_channel=FakeHumanChannel(),
        executor=FakeExecutor(),
    )
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="clarifying",
        iteration=1,
        max_iterations=5,
        persona="missing",
        created_at="t",
        updated_at="t",
    )
    with patch("mltgnt.loops.engine.load_persona", side_effect=FileNotFoundError("missing")):
        assert engine._resolve_llm_engine_model(state) == ("cursor", "auto")
        assert engine._resolve_subtask_engine_model(state) == ("cursor", "auto")


def _llm_call_error(message: str = "llm failed") -> prompts.LlmCallError:
    from tests.loops.fakes import FakeLLMResult

    # raw_output に非 JSON ネイティブを混ぜても記録経路が落ちないこと
    trace = prompts.LlmTrace(
        input="prompt",
        raw_output=FakeLLMResult(stdout="bad", stderr="boom", returncode=1),  # type: ignore[arg-type]
        parsed=None,
        reasoning="",
        config={"engine": "claude", "model": "m"},
        metadata={"retry": True},
        uncertain_flag=False,
        error=message,
    )
    return prompts.LlmCallError(message, trace)


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_clarify")
def test_clarify_llm_errors_fail_after_three_consecutive_ticks(mock_clarify, mock_persona, tmp_path):
    persona = MagicMock()
    persona.format_prompt.side_effect = lambda x, **_: x
    mock_persona.return_value = persona
    mock_clarify.side_effect = _llm_call_error("clarify failed")

    engine = _engine(tmp_path)
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
        created_at="t",
        updated_at="t",
    )
    store.save_state(_config(tmp_path).state_dir, state)

    engine.tick()
    engine.tick()
    engine.tick()

    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state is not None
    assert state.status == "failed"
    assert state.consecutive_errors == 3
    events = store.read_events(_config(tmp_path).state_dir, "loop1")
    llm_errors = [e for e in events if e["event"] == "llm_error"]
    assert len(llm_errors) == 3
    assert all(isinstance(e["data"]["error"], str) for e in llm_errors)


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_decompose")
def test_decompose_llm_errors_fail_after_three_consecutive_ticks(mock_decompose, mock_persona, tmp_path):
    persona = MagicMock()
    persona.format_prompt.side_effect = lambda x, **_: x
    mock_persona.return_value = persona
    mock_decompose.side_effect = _llm_call_error("decompose failed")

    engine = _engine(tmp_path)
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="decomposing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        created_at="t",
        updated_at="t",
    )
    store.save_state(_config(tmp_path).state_dir, state)

    engine.tick()
    engine.tick()
    engine.tick()

    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state is not None
    assert state.status == "failed"
    assert state.consecutive_errors == 3


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_evaluate")
def test_evaluate_llm_errors_fail_after_three_consecutive_ticks(mock_evaluate, mock_persona, tmp_path):
    persona = MagicMock()
    persona.format_prompt.side_effect = lambda x, **_: x
    mock_persona.return_value = persona
    mock_evaluate.side_effect = _llm_call_error("evaluate failed")

    engine = _engine(tmp_path)
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="evaluating",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        subtasks=[Subtask(id="s1", title="S1", kind="auto", prompt="p", status="success", result="ok")],
        created_at="t",
        updated_at="t",
    )
    store.save_state(_config(tmp_path).state_dir, state)

    engine.tick()
    engine.tick()
    engine.tick()

    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state is not None
    assert state.status == "failed"
    assert state.consecutive_errors == 3


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_clarify")
def test_record_llm_error_increments_once_per_tick(mock_clarify, mock_persona, tmp_path):
    """1 tick の LlmCallError で consecutive_errors は 1 だけ増える。"""
    persona = MagicMock()
    persona.format_prompt.side_effect = lambda x, **_: x
    mock_persona.return_value = persona
    mock_clarify.side_effect = _llm_call_error("once")

    engine = _engine(tmp_path)
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
        created_at="t",
        updated_at="t",
    )
    store.save_state(_config(tmp_path).state_dir, state)

    engine.tick()

    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state is not None
    assert state.consecutive_errors == 1
    assert state.status == "clarifying"


def test_record_llm_error_increments_once_when_event_write_fails(tmp_path):
    """イベント記録失敗時も consecutive_errors を二重加算しない。"""
    engine = _engine(tmp_path)
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
        created_at="t",
        updated_at="t",
    )

    with patch("mltgnt.loops.engine.store.append_event", side_effect=OSError("disk full")):
        engine._record_llm_error(state, "clarify", RuntimeError("llm failed"))

    assert state.consecutive_errors == 1
    assert state.status == "clarifying"


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_clarify")
def test_start_loop_inherits_thread_without_open_thread(mock_clarify, mock_persona, tmp_path):
    from mltgnt.interfaces.loops import HumanThreadRef

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
    thread = HumanThreadRef(channel_id="C1", thread_ts="123.456")
    engine.start_loop(_objective(), thread=thread)

    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state.thread == thread
    assert channel.open_thread_calls == 0

    engine.tick()
    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state.status == "awaiting_answer"
    assert channel.open_thread_calls == 0
    assert channel.asks[0]["thread"] == thread


def test_start_loop_without_thread_still_opens(tmp_path):
    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel)
    engine.start_loop(_objective())
    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state.thread is None
    assert channel.open_thread_calls == 0


def test_consecutive_errors_failed_closes_thread_once(tmp_path):
    """consecutive_errors が上限に達して failed へ遷移したら close_thread が 1 回だけ呼ばれる。"""
    executor = FakeExecutor()
    executor.poll = MagicMock(side_effect=RuntimeError("poll unavailable"))
    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel, executor=executor)
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="executing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        subtasks=[
            Subtask(
                id="s1",
                title="S1",
                kind="auto",
                prompt="do it",
                status="running",
                submission=StepSubmission(
                    uuid="u1",
                    result_filename="r1.md",
                    submitted_at="2026-08-20T12:00:00+09:00",
                    reused=False,
                ),
            )
        ],
        current_subtask_id="s1",
        created_at="t",
        updated_at="t",
    )
    store.save_state(_config(tmp_path).state_dir, state)

    engine.tick()
    engine.tick()
    engine.tick()

    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state.status == "failed"
    assert len(channel.closes) == 1
    assert channel.closes[0]["loop_id"] == "loop1"


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_evaluate")
def test_terminal_paths_each_close_thread_once(mock_evaluate, mock_persona, tmp_path):
    """done / cancelled / failed の 3 終端それぞれで close_thread がちょうど 1 回。"""
    persona = MagicMock()
    persona.format_prompt.side_effect = lambda x, **_: x
    mock_persona.return_value = persona

    # --- done ---
    mock_evaluate.return_value = (
        prompts.EvaluateResponse(
            achieved=True,
            score=100,
            summary="ok",
            next_focus="",
            reasoning="",
            uncertain_flag=False,
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    channel_done = FakeHumanChannel()
    engine_done = _engine(tmp_path, channel=channel_done)
    store.save_state(
        _config(tmp_path).state_dir,
        LoopState(
            loop_id="done1",
            objective_path="/tmp/x.md",
            objective_hash="h",
            title="T",
            body="body",
            status="evaluating",
            iteration=1,
            max_iterations=5,
            persona="mizuho",
            thread=_thread(),
            subtasks=[
                Subtask(id="s1", title="S1", kind="auto", prompt="p", status="success", result="ok")
            ],
            created_at="t",
            updated_at="t",
        ),
    )
    engine_done.tick()
    assert store.load_state(_config(tmp_path).state_dir, "done1").status == "done"
    assert len(channel_done.closes) == 1

    # --- cancelled ---
    channel_cancel = FakeHumanChannel()
    engine_cancel = _engine(tmp_path, channel=channel_cancel)
    store.save_state(
        _config(tmp_path).state_dir,
        LoopState(
            loop_id="cancel1",
            objective_path="/tmp/x.md",
            objective_hash="h",
            title="T",
            body="body",
            status="executing",
            iteration=1,
            max_iterations=5,
            persona="mizuho",
            thread=_thread(),
            created_at="t",
            updated_at="t",
        ),
    )
    inbox = store._inbox_dir(_config(tmp_path).state_dir, "cancel1")
    inbox.mkdir(parents=True)
    (inbox / "001-c.json").write_text(
        json.dumps(
            {
                "kind": "cancel",
                "message_id": "c1",
                "question_id": "",
                "text": "",
                "received_at": "2026-08-20T12:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )
    engine_cancel.tick()
    assert store.load_state(_config(tmp_path).state_dir, "cancel1").status == "cancelled"
    assert len(channel_cancel.closes) == 1

    # --- failed (max iterations) ---
    mock_evaluate.return_value = (
        prompts.EvaluateResponse(
            achieved=False,
            score=10,
            summary="not yet",
            next_focus="retry",
            reasoning="",
            uncertain_flag=False,
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    channel_fail = FakeHumanChannel()
    engine_fail = _engine(tmp_path, channel=channel_fail)
    store.save_state(
        _config(tmp_path).state_dir,
        LoopState(
            loop_id="fail1",
            objective_path="/tmp/x.md",
            objective_hash="h",
            title="T",
            body="body",
            status="evaluating",
            iteration=5,
            max_iterations=5,
            persona="mizuho",
            thread=_thread(),
            subtasks=[
                Subtask(id="s1", title="S1", kind="auto", prompt="p", status="success", result="ok")
            ],
            created_at="t",
            updated_at="t",
        ),
    )
    engine_fail.tick()
    assert store.load_state(_config(tmp_path).state_dir, "fail1").status == "failed"
    assert len(channel_fail.closes) == 1


def test_comment_inbox_appends_clarification_context(tmp_path):
    """kind=comment を 1 件処理すると clarification_context に追記し comment_received を記録する。"""
    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel)
    store.save_state(
        _config(tmp_path).state_dir,
        LoopState(
            loop_id="loop1",
            objective_path="/tmp/x.md",
            objective_hash="h",
            title="T",
            body="body",
            status="decomposing",
            iteration=1,
            max_iterations=5,
            persona="mizuho",
            thread=_thread(),
            plan_approval=False,
            created_at="t",
            updated_at="t",
        ),
    )
    inbox = store._inbox_dir(_config(tmp_path).state_dir, "loop1")
    inbox.mkdir(parents=True)
    (inbox / "001-comment.json").write_text(
        json.dumps(
            {
                "kind": "comment",
                "message_id": "cm1",
                "question_id": "",
                "text": "締切は金曜",
                "received_at": "2026-08-20T12:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )

    with patch("mltgnt.loops.engine.load_persona") as mock_persona:
        persona = MagicMock()
        persona.format_prompt.side_effect = lambda x, **_: x
        persona.fm.engine = ""
        persona.fm.model = ""
        mock_persona.return_value = persona
        with patch("mltgnt.loops.engine.prompts.run_decompose") as mock_decompose:
            mock_decompose.return_value = (
                prompts.DecomposeResponse(
                    subtasks=[
                        prompts.DecomposeSubtask(id="s1", title="T", kind="auto", prompt="p")
                    ],
                    reasoning="",
                    uncertain_flag=False,
                ),
                prompts.LlmTrace("", "", {}, "", {}, {}, False),
            )
            engine.tick()

    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state.clarification_context == ["補足: 締切は金曜"]
    events = store.read_events(_config(tmp_path).state_dir, "loop1")
    comments = [e for e in events if e["event"] == "comment_received"]
    assert len(comments) == 1
    assert comments[0]["data"]["text"] == "締切は金曜"
    assert "cm1" in store.list_consumed_message_ids(_config(tmp_path).state_dir, "loop1")


def test_comment_inbox_not_double_consumed(tmp_path):
    """同一 comment を再処理しても clarification_context に重複追記しない。"""
    engine = _engine(tmp_path)
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
        thread=_thread(),
        created_at="t",
        updated_at="t",
    )
    store.save_state(_config(tmp_path).state_dir, state)
    inbox = store._inbox_dir(_config(tmp_path).state_dir, "loop1")
    inbox.mkdir(parents=True)
    (inbox / "001-comment.json").write_text(
        json.dumps(
            {
                "kind": "comment",
                "message_id": "cm1",
                "question_id": "",
                "text": "補足メモ",
                "received_at": "2026-08-20T12:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )

    with patch("mltgnt.loops.engine.load_persona") as mock_persona:
        persona = MagicMock()
        persona.format_prompt.side_effect = lambda x, **_: x
        persona.fm.engine = ""
        persona.fm.model = ""
        mock_persona.return_value = persona
        with patch("mltgnt.loops.engine.prompts.run_clarify") as mock_clarify:
            mock_clarify.return_value = (
                prompts.ClarifyResponse(
                    clear=True, question="", reason="", reasoning="", uncertain_flag=False
                ),
                prompts.LlmTrace("", "", {}, "", {}, {}, False),
            )
            engine.tick()
            # 再 tick（consumed 済みなので追記されない）。status は decomposing になっている
            engine.tick()

    state = store.load_state(_config(tmp_path).state_dir, "loop1")
    assert state.clarification_context == ["補足: 補足メモ"]
    events = store.read_events(_config(tmp_path).state_dir, "loop1")
    assert len([e for e in events if e["event"] == "comment_received"]) == 1


def test_start_loop_initializes_deliverable_without_touching_objective(tmp_path):
    obj_path = tmp_path / "obj.md"
    obj_path.write_text("alpha\nβ", encoding="utf-8")
    original = obj_path.read_text(encoding="utf-8")
    engine = _engine(tmp_path)
    obj = Objective(
        loop_id="loop1",
        title="Title",
        body="alpha\nβ",
        agent="mizuho",
        max_iterations=5,
        status="active",
        path=obj_path,
        content_hash="hash1",
    )
    engine.start_loop(obj)
    deliverable = store.deliverable_path(_config(tmp_path).state_dir, "loop1")
    assert deliverable.read_text(encoding="utf-8") == "alpha\nβ\n"
    assert obj_path.read_text(encoding="utf-8") == original
    events = store.read_events(_config(tmp_path).state_dir, "loop1")
    assert any(e["event"] == "deliverable_updated" for e in events)


def test_config_rejects_non_positive_summary_limits(tmp_path):
    with pytest.raises(ValueError, match="deliverable_excerpt_chars"):
        LoopsConfig(
            objectives_dir=tmp_path / "o",
            state_dir=tmp_path / "s",
            status_dir=tmp_path / "st",
            jobs_dir=tmp_path / "j",
            exec_done_dir=tmp_path / "j" / "done",
            persona_dir=tmp_path / "p",
            default_persona="mizuho",
            fallback_channel="C",
            deliverable_excerpt_chars=0,
        )
    with pytest.raises(ValueError, match="result_summary_chars"):
        LoopsConfig(
            objectives_dir=tmp_path / "o",
            state_dir=tmp_path / "s",
            status_dir=tmp_path / "st",
            jobs_dir=tmp_path / "j",
            exec_done_dir=tmp_path / "j" / "done",
            persona_dir=tmp_path / "p",
            default_persona="mizuho",
            fallback_channel="C",
            result_summary_chars=0,
        )


def test_subtask_from_dict_fills_missing_summary_fields():
    data = {
        "id": "s1",
        "title": "T",
        "kind": "auto",
        "prompt": "p",
        "status": "success",
        "result": "full stdout",
        "submission": {
            "uuid": "u1",
            "result_filename": "r1.md",
            "submitted_at": "2026-08-20T12:00:00+09:00",
            "reused": False,
        },
    }
    st = Subtask.from_dict(data)
    assert st.result_summary == ""
    assert st.result_filename == ""
    assert st.result == "full stdout"
    assert st.submission is not None
    assert st.submission.result_filename == "r1.md"


@patch("mltgnt.loops.engine.load_persona")
def test_auto_submit_prompt_includes_deliverable_contract(mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops()
    executor = FakeExecutor()
    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel, executor=executor)
    cfg = _config(tmp_path)
    store.initialize_deliverable(cfg.state_dir, "loop1", "seed body")
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="executing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        subtasks=[Subtask(id="s1", title="S1", kind="auto", prompt="do it", status="pending")],
        current_subtask_id="s1",
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg.state_dir, state)
    engine.tick()

    prompt = executor.submit_kwargs[0]["prompt"]
    path = str(store.deliverable_path(cfg.state_dir, "loop1"))
    assert path in prompt
    assert "Edit this file directly" in prompt
    assert "Do not create new deliverable" in prompt
    assert "3-5 line" in prompt
    assert "seed body" in prompt
    events = store.read_events(cfg.state_dir, "loop1")
    submitted = [e for e in events if e["event"] == "subtask_submitted"]
    assert len(submitted) == 1
    assert submitted[0]["data"]["result_filename"] == "result-1.md"


@patch("mltgnt.loops.engine.load_persona")
def test_poll_success_persists_summary_and_events(mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops()
    cfg = _config(tmp_path)
    long_out = "X" * 1500
    executor = FakeExecutor()
    executor.poll_results["u1"] = StepPoll(status="success", content=long_out)
    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel, executor=executor)
    store.initialize_deliverable(cfg.state_dir, "loop1", "seed")
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="executing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        subtasks=[
            Subtask(
                id="s1",
                title="S1",
                kind="auto",
                prompt="do it",
                status="running",
                submission=StepSubmission(
                    uuid="u1",
                    result_filename="r1.md",
                    submitted_at="2026-08-20T12:00:00+09:00",
                    reused=False,
                ),
            )
        ],
        current_subtask_id="s1",
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg.state_dir, state)
    engine.tick()

    state = store.load_state(cfg.state_dir, "loop1")
    assert state is not None
    st = state.subtasks[0]
    assert st.result == long_out
    assert st.result_summary == long_out[:1000]
    assert st.result_filename == "r1.md"
    events = store.read_events(cfg.state_dir, "loop1")
    assert any(e["event"] == "subtask_done" for e in events)
    assert any(e["event"] == "deliverable_updated" for e in events)
    assert len(channel.progress_posts) == 1
    assert state.status == "evaluating"


@patch("mltgnt.loops.engine.load_persona")
def test_poll_failed_and_timeout_record_summary(mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops()
    cfg = _config(tmp_path)
    executor = FakeExecutor()
    executor.poll_results["u-fail"] = StepPoll(status="failed_exit", content="boom")
    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel, executor=executor)
    store.initialize_deliverable(cfg.state_dir, "loop1", "seed")
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="executing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        subtasks=[
            Subtask(
                id="s1",
                title="S1",
                kind="auto",
                prompt="do it",
                status="running",
                submission=StepSubmission(
                    uuid="u-fail",
                    result_filename="r-fail.md",
                    submitted_at="2026-08-20T12:00:00+09:00",
                    reused=False,
                ),
            )
        ],
        current_subtask_id="s1",
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg.state_dir, state)
    engine.tick()
    state = store.load_state(cfg.state_dir, "loop1")
    assert state.subtasks[0].status == "failed"
    assert state.subtasks[0].result_summary == "boom"
    assert any(e["event"] == "subtask_done" for e in store.read_events(cfg.state_dir, "loop1"))

    # timeout path
    executor2 = FakeExecutor()
    engine2 = _engine(tmp_path / "t2", channel=FakeHumanChannel(), executor=executor2)
    cfg2 = _config(tmp_path / "t2")
    store.initialize_deliverable(cfg2.state_dir, "loop2", "seed")
    state2 = LoopState(
        loop_id="loop2",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="executing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        subtasks=[
            Subtask(
                id="s1",
                title="S1",
                kind="auto",
                prompt="do it",
                status="running",
                submission=StepSubmission(
                    uuid="u-to",
                    result_filename="r-to.md",
                    submitted_at="2000-01-01T00:00:00+09:00",
                    reused=False,
                ),
            )
        ],
        current_subtask_id="s1",
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg2.state_dir, state2)
    engine2.tick()
    state2 = store.load_state(cfg2.state_dir, "loop2")
    assert state2.subtasks[0].status == "failed"
    assert "timeout" in state2.subtasks[0].result_summary
    assert any(e["event"] == "subtask_done" for e in store.read_events(cfg2.state_dir, "loop2"))


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_evaluate")
def test_evaluate_uses_result_summary_and_deliverable(mock_evaluate, mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops()
    mock_evaluate.return_value = (
        prompts.EvaluateResponse(
            achieved=True,
            score=100,
            summary="done",
            next_focus="",
            reasoning="",
            uncertain_flag=False,
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    cfg = _config(tmp_path)
    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel)
    store.initialize_deliverable(cfg.state_dir, "loop1", "iteration1 body content")
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="objective body",
        status="evaluating",
        iteration=2,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        subtasks=[
            Subtask(
                id="s1",
                title="S1",
                kind="auto",
                prompt="p",
                status="success",
                result="raw long",
                result_summary="short summary",
            )
        ],
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg.state_dir, state)
    engine.tick()

    instruction = mock_evaluate.call_args.args[0]
    assert "short summary" in instruction
    assert "iteration1 body content" in instruction


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_decompose")
def test_progress_notify_plan_and_toggle(mock_decompose, mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops()
    mock_decompose.return_value = (
        prompts.DecomposeResponse(
            subtasks=[
                prompts.DecomposeSubtask(id="s1", title="One", kind="auto", prompt="p"),
                prompts.DecomposeSubtask(
                    id="s2", title="Two", kind="human", prompt="q", depends=("s1",)
                ),
            ],
            reasoning="",
            uncertain_flag=False,
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
        status="decomposing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        plan_approval=False,
        created_at="t",
        updated_at="t",
    )
    store.save_state(_config(tmp_path).state_dir, state)
    engine.tick()
    assert len(channel.progress_posts) == 1
    assert "One" in channel.progress_posts[0]["text"]

    channel2 = FakeHumanChannel()
    cfg2 = _config(tmp_path / "off")
    cfg_off = LoopsConfig(
        objectives_dir=cfg2.objectives_dir,
        state_dir=cfg2.state_dir,
        status_dir=cfg2.status_dir,
        jobs_dir=cfg2.jobs_dir,
        exec_done_dir=cfg2.exec_done_dir,
        persona_dir=cfg2.persona_dir,
        default_persona="mizuho",
        fallback_channel="C-fallback",
        progress_notify=False,
    )
    engine_off = LoopsEngine(
        config=cfg_off,
        human_channel=channel2,
        executor=FakeExecutor(),
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
    )
    state2 = LoopState(
        loop_id="loop2",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="decomposing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        plan_approval=False,
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg_off.state_dir, state2)
    engine_off.tick()
    assert channel2.progress_posts == []


@patch("mltgnt.loops.engine.load_persona")
def test_human_deliverable_posts_and_idempotent_retick(mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops()
    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel)
    cfg = _config(tmp_path)
    store.initialize_deliverable(cfg.state_dir, "loop1", "seed")
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="executing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        subtasks=[Subtask(id="h1", title="Review", kind="human", prompt="check please", status="pending")],
        current_subtask_id="h1",
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg.state_dir, state)
    engine.tick()
    assert len(channel.deliverable_posts) == 1
    assert channel.deliverable_posts[0]["deliverable_path"] == str(
        store.deliverable_path(cfg.state_dir, "loop1")
    )
    start_eid = channel.deliverable_posts[0]["event_id"]
    engine.tick()
    assert len(channel.deliverable_posts) == 1

    inbox = store._inbox_dir(cfg.state_dir, "loop1")
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "001-a.json").write_text(
        json.dumps(
            {
                "kind": "answer",
                "message_id": "m1",
                "question_id": "human-1-h1",
                "text": "LGTM",
                "received_at": "2026-08-20T12:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )
    engine.tick()
    assert len(channel.deliverable_posts) == 2
    assert channel.deliverable_posts[1]["summary"] == "LGTM"
    done_eid = channel.deliverable_posts[1]["event_id"]
    assert start_eid != done_eid
    engine.tick()
    assert len(channel.deliverable_posts) == 2
    state = store.load_state(cfg.state_dir, "loop1")
    assert state.subtasks[0].result_summary == "LGTM"
    assert state.subtasks[0].result_filename == ""


@patch("mltgnt.loops.engine.load_persona")
def test_channel_false_or_exception_keeps_subtask_success(mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops()
    cfg = _config(tmp_path)
    executor = FakeExecutor()
    executor.poll_results["u1"] = StepPoll(status="success", content="ok")
    channel = FakeHumanChannel(post_progress_result=False)
    engine = _engine(tmp_path, channel=channel, executor=executor)
    store.initialize_deliverable(cfg.state_dir, "loop1", "seed")
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="executing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        subtasks=[
            Subtask(
                id="s1",
                title="S1",
                kind="auto",
                prompt="do it",
                status="running",
                submission=StepSubmission(
                    uuid="u1",
                    result_filename="r1.md",
                    submitted_at="2026-08-20T12:00:00+09:00",
                    reused=False,
                ),
            )
        ],
        current_subtask_id="s1",
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg.state_dir, state)
    engine.tick()
    state = store.load_state(cfg.state_dir, "loop1")
    assert state.subtasks[0].status == "success"
    assert state.status == "evaluating"
    assert any(
        e["event"] == "channel_error" and e["data"]["action"] == "post_progress"
        for e in store.read_events(cfg.state_dir, "loop1")
    )

    channel_exc = FakeHumanChannel(post_progress_exc=RuntimeError("slack down"))
    executor2 = FakeExecutor()
    executor2.poll_results["u2"] = StepPoll(status="success", content="ok2")
    engine2 = _engine(tmp_path / "exc", channel=channel_exc, executor=executor2)
    cfg2 = _config(tmp_path / "exc")
    store.initialize_deliverable(cfg2.state_dir, "loop2", "seed")
    state2 = LoopState(
        loop_id="loop2",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="executing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        subtasks=[
            Subtask(
                id="s1",
                title="S1",
                kind="auto",
                prompt="do it",
                status="running",
                submission=StepSubmission(
                    uuid="u2",
                    result_filename="r2.md",
                    submitted_at="2026-08-20T12:00:00+09:00",
                    reused=False,
                ),
            )
        ],
        current_subtask_id="s1",
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg2.state_dir, state2)
    engine2.tick()
    state2 = store.load_state(cfg2.state_dir, "loop2")
    assert state2.subtasks[0].status == "success"
    assert state2.status == "evaluating"


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_clarify")
@patch("mltgnt.loops.engine.prompts.run_decompose")
@patch("mltgnt.loops.engine.prompts.run_evaluate")
def test_state_change_and_observability_events(
    mock_evaluate, mock_decompose, mock_clarify, mock_persona, tmp_path
):
    mock_persona.return_value = _persona_with_ops()
    mock_clarify.return_value = (
        prompts.ClarifyResponse(
            clear=False, question="いつ？", reason="", reasoning="", uncertain_flag=False
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    mock_decompose.return_value = (
        prompts.DecomposeResponse(
            subtasks=[
                prompts.DecomposeSubtask(id="s1", title="Auto", kind="auto", prompt="work"),
                prompts.DecomposeSubtask(
                    id="h1", title="Human", kind="human", prompt="review", depends=("s1",)
                ),
            ],
            reasoning="",
            uncertain_flag=False,
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    mock_evaluate.return_value = (
        prompts.EvaluateResponse(
            achieved=True,
            score=100,
            summary="done",
            next_focus="",
            reasoning="",
            uncertain_flag=False,
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    executor = FakeExecutor()
    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel, executor=executor)
    cfg = _config(tmp_path)
    engine.start_loop(_objective())

    # clarifying -> awaiting_answer
    engine.tick()
    state = store.load_state(cfg.state_dir, "loop1")
    assert state.status == "awaiting_answer"
    qid = channel.asks[0]["question_id"]

    inbox = store._inbox_dir(cfg.state_dir, "loop1")
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "001-a.json").write_text(
        json.dumps(
            {
                "kind": "answer",
                "message_id": "m1",
                "question_id": qid,
                "text": "来月",
                "received_at": "2026-08-20T12:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )
    # awaiting_answer -> clarifying
    engine.tick()
    mock_clarify.return_value = (
        prompts.ClarifyResponse(
            clear=True, question="", reason="", reasoning="", uncertain_flag=False
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    # clarifying -> decomposing
    engine.tick()
    # decomposing -> executing
    engine.tick()
    # submit auto
    engine.tick()
    executor.poll_results["uuid-1"] = StepPoll(status="success", content="auto done")
    # poll success -> next human
    engine.tick()
    # human ask -> awaiting_human
    engine.tick()
    (inbox / "002-h.json").write_text(
        json.dumps(
            {
                "kind": "answer",
                "message_id": "m2",
                "question_id": "human-1-h1",
                "text": "approved",
                "received_at": "2026-08-20T12:05:00+09:00",
            }
        ),
        encoding="utf-8",
    )
    # human done -> evaluating
    engine.tick()
    # evaluating -> done
    engine.tick()

    events = store.read_events(cfg.state_dir, "loop1")
    changes = [e for e in events if e["event"] == "state_change"]
    pairs = [(e["data"]["from"], e["data"]["to"]) for e in changes]
    assert ("clarifying", "awaiting_answer") in pairs
    assert ("awaiting_answer", "clarifying") in pairs
    assert ("clarifying", "decomposing") in pairs
    assert ("decomposing", "executing") in pairs
    assert ("executing", "awaiting_human") in pairs
    assert ("awaiting_human", "executing") in pairs or ("awaiting_human", "evaluating") in pairs
    assert any(e["data"]["to"] == "evaluating" for e in changes)
    assert any(e["data"]["to"] == "done" for e in changes)
    assert all("reason" in e["data"] for e in changes)

    assert any(e["event"] == "question_asked" for e in events)
    assert any(e["event"] == "subtask_submitted" for e in events)
    assert any(e["event"] == "subtask_done" for e in events)
    assert any(e["event"] == "deliverable_updated" for e in events)

    # re-tick terminal: no duplicate state_change to done
    done_count_before = len([e for e in events if e["event"] == "state_change" and e["data"]["to"] == "done"])
    engine.tick()
    events_after = store.read_events(cfg.state_dir, "loop1")
    done_count_after = len(
        [e for e in events_after if e["event"] == "state_change" and e["data"]["to"] == "done"]
    )
    assert done_count_after == done_count_before


@dataclass
class FakeConditionEvaluator:
    verdicts: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def evaluate(self, condition, *, previous_token):
        self.calls.append({"condition": dict(condition), "previous_token": previous_token})
        if not self.verdicts:
            from mltgnt.interfaces.loops import WatchVerdict
            return WatchVerdict(status="pending", detail="waiting")
        return self.verdicts.pop(0)


def _executing_watch_state(*, watch_kwargs=None, auto_kwargs=None) -> LoopState:
    watch = dict(
        id="w1",
        title="Watch",
        kind="watch",
        prompt="",
        status="pending",
        condition={"type": "path_exists", "path": "flag"},
        depends=[],
        timeout_sec=100,
        poll_interval_sec=30,
    )
    if watch_kwargs:
        watch.update(watch_kwargs)
    auto = dict(
        id="a1",
        title="Auto",
        kind="auto",
        prompt="work",
        status="pending",
        depends=[],
    )
    if auto_kwargs:
        auto.update(auto_kwargs)
    return LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="executing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        plan_approval=False,
        subtasks=[Subtask(**watch), Subtask(**auto)],
        created_at="t",
        updated_at="t",
    )


@patch("mltgnt.loops.engine.load_persona")
def test_watch_and_auto_parallel_but_single_auto_submit(mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops()
    root = tmp_path / "watch_root"
    root.mkdir()
    (root / "flag").write_text("1", encoding="utf-8")
    cfg = _config(tmp_path)
    # rebuild config with watch_root
    cfg = LoopsConfig(
        objectives_dir=cfg.objectives_dir,
        state_dir=cfg.state_dir,
        status_dir=cfg.status_dir,
        jobs_dir=cfg.jobs_dir,
        exec_done_dir=cfg.exec_done_dir,
        persona_dir=cfg.persona_dir,
        default_persona="mizuho",
        fallback_channel="C-fallback",
        watch_root=root,
    )
    executor = FakeExecutor()
    engine = LoopsEngine(
        config=cfg,
        human_channel=FakeHumanChannel(),
        executor=executor,
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
    )
    state = _executing_watch_state()
    store.save_state(cfg.state_dir, state)
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime(2026, 8, 20, 12, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    engine.tick(now=now)
    state = store.load_state(cfg.state_dir, "loop1")
    assert state.subtasks[0].status == "success"  # path exists
    assert state.subtasks[1].status == "running"
    assert len(executor.submit_calls) == 1

    # second auto pending should not submit while first running
    state.subtasks.append(
        Subtask(id="a2", title="A2", kind="auto", prompt="more", status="pending", depends=[])
    )
    store.save_state(cfg.state_dir, state)
    engine.tick(now=now)
    assert len(executor.submit_calls) == 1


@patch("mltgnt.loops.engine.load_persona")
def test_watch_poll_interval_timeout_and_event_dedup(mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops()
    from mltgnt.interfaces.loops import WatchVerdict
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    fake = FakeConditionEvaluator(
        verdicts=[
            WatchVerdict(status="pending", detail="wait", observed_token="t1"),
            WatchVerdict(status="pending", detail="wait", observed_token="t1"),
        ]
    )
    cfg = _config(tmp_path)
    engine = LoopsEngine(
        config=cfg,
        human_channel=FakeHumanChannel(),
        executor=FakeExecutor(),
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
        condition_evaluator=fake,
    )
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="executing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        plan_approval=False,
        subtasks=[
            Subtask(
                id="w1",
                title="W",
                kind="watch",
                prompt="",
                condition={"type": "issue_label", "number": 1},
                depends=[],
                timeout_sec=100,
                poll_interval_sec=30,
            )
        ],
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg.state_dir, state)
    t0 = datetime(2026, 8, 20, 12, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    engine.tick(now=t0)
    assert len(fake.calls) == 1
    events = [e for e in store.read_events(cfg.state_dir, "loop1") if e["event"] == "watch_polled"]
    assert len(events) == 1

    # within poll interval: no re-call
    engine.tick(now=t0 + timedelta(seconds=10))
    assert len(fake.calls) == 1
    events = [e for e in store.read_events(cfg.state_dir, "loop1") if e["event"] == "watch_polled"]
    assert len(events) == 1

    # after interval, same pending → no new event
    engine.tick(now=t0 + timedelta(seconds=30))
    assert len(fake.calls) == 2
    events = [e for e in store.read_events(cfg.state_dir, "loop1") if e["event"] == "watch_polled"]
    assert len(events) == 1

    # timeout at elapsed == timeout_sec
    state = store.load_state(cfg.state_dir, "loop1")
    store.save_state(cfg.state_dir, state)
    engine.tick(now=t0 + timedelta(seconds=100))
    state = store.load_state(cfg.state_dir, "loop1")
    assert state.subtasks[0].status == "failed"
    assert state.status == "replanning"


@patch("mltgnt.loops.engine.load_persona")
def test_host_evaluator_paths_and_failures(mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops()
    from mltgnt.interfaces.loops import WatchVerdict
    from datetime import datetime
    from zoneinfo import ZoneInfo

    cfg = _config(tmp_path)
    t0 = datetime(2026, 8, 20, 12, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))

    def _watch_state():
        return LoopState(
            loop_id="loop1",
            objective_path="/tmp/x.md",
            objective_hash="h",
            title="T",
            body="body",
            status="executing",
            iteration=1,
            max_iterations=5,
            persona="mizuho",
            thread=_thread(),
            plan_approval=False,
            subtasks=[
                Subtask(
                    id="w1",
                    title="W",
                    kind="watch",
                    prompt="",
                    condition={"type": "issue_label", "number": 1},
                    depends=[],
                    timeout_sec=1000,
                    poll_interval_sec=1,
                )
            ],
            created_at="t",
            updated_at="t",
        )

    # satisfied via fake
    fake = FakeConditionEvaluator(
        verdicts=[WatchVerdict(status="satisfied", detail="labeled")]
    )
    engine = LoopsEngine(
        config=cfg,
        human_channel=FakeHumanChannel(),
        executor=FakeExecutor(),
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
        condition_evaluator=fake,
    )
    store.save_state(cfg.state_dir, _watch_state())
    engine.tick(now=t0)
    assert store.load_state(cfg.state_dir, "loop1").subtasks[0].status == "success"

    # no evaluator
    engine2 = LoopsEngine(
        config=_config(tmp_path / "n"),
        human_channel=FakeHumanChannel(),
        executor=FakeExecutor(),
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
    )
    store.save_state(_config(tmp_path / "n").state_dir, _watch_state())
    engine2.tick(now=t0)
    st = store.load_state(_config(tmp_path / "n").state_dir, "loop1")
    assert st.subtasks[0].status == "failed"
    assert st.status == "replanning"

    # None / exception / bad status
    class BadEval:
        def __init__(self, mode):
            self.mode = mode

        def evaluate(self, condition, *, previous_token):
            if self.mode == "none":
                return None
            if self.mode == "exc":
                raise RuntimeError("boom")
            return WatchVerdict(status="weird", detail="x")  # type: ignore[arg-type]

    for mode, path in [("none", "none"), ("exc", "exc"), ("bad", "bad")]:
        p = tmp_path / path
        eng = LoopsEngine(
            config=_config(p),
            human_channel=FakeHumanChannel(),
            executor=FakeExecutor(),
            objective_exists=lambda _: True,
            objective_cancelled=lambda _: False,
            objective_hash_changed=lambda *_: False,
            condition_evaluator=BadEval(mode),
        )
        store.save_state(_config(p).state_dir, _watch_state())
        eng.tick(now=t0)
        st = store.load_state(_config(p).state_dir, "loop1")
        assert st.subtasks[0].status == "failed"
        assert st.status == "replanning"


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_replan")
def test_watch_fail_replan_and_limit(mock_replan, mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops()
    from datetime import datetime
    from zoneinfo import ZoneInfo

    cfg = _config(tmp_path)
    channel = FakeHumanChannel()
    engine = LoopsEngine(
        config=cfg,
        human_channel=channel,
        executor=FakeExecutor(),
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
    )
    t0 = datetime(2026, 8, 20, 12, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="replanning",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        plan_approval=False,
        replan_count=0,
        replan_trigger_subtask_id="w1",
        replan_feedback="watch failed",
        subtasks=[
            Subtask(id="s1", title="S", kind="auto", prompt="p", status="success", depends=[]),
            Subtask(
                id="w1",
                title="W",
                kind="watch",
                prompt="",
                status="failed",
                depends=["s1"],
                condition={"type": "issue_label"},
                result_summary="watch failed",
            ),
        ],
        created_at="t",
        updated_at="t",
    )
    mock_replan.return_value = (
        prompts.ReplanResponse(
            keep=("s1",),
            add=(
                prompts.DecomposeSubtask(
                    id="a2", title="Investigate", kind="auto", prompt="look", depends=()
                ),
            ),
            reason="retry",
            reasoning="",
            uncertain_flag=False,
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    store.save_state(cfg.state_dir, state)
    engine.tick(now=t0)
    state = store.load_state(cfg.state_dir, "loop1")
    assert state.status == "executing"
    assert state.replan_count == 1
    assert [s.id for s in state.subtasks] == ["s1", "a2"]
    assert any(e["event"] == "replan_triggered" for e in store.read_events(cfg.state_dir, "loop1"))

    # hit limit → evaluating without LLM
    state.replan_count = 3
    state.status = "replanning"
    state.replan_trigger_subtask_id = "a2"
    state.replan_feedback = "again"
    store.save_state(cfg.state_dir, state)
    mock_replan.reset_mock()
    engine.tick(now=t0)
    state = store.load_state(cfg.state_dir, "loop1")
    assert state.status == "evaluating"
    mock_replan.assert_not_called()


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_decompose")
def test_plan_approval_gate_and_phrases(mock_decompose, mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops()
    mock_decompose.return_value = (
        prompts.DecomposeResponse(
            subtasks=[prompts.DecomposeSubtask(id="s1", title="One", kind="auto", prompt="p")],
            reasoning="",
            uncertain_flag=False,
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel)
    cfg = _config(tmp_path)
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="decomposing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        plan_approval=True,
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg.state_dir, state)
    engine.tick()
    state = store.load_state(cfg.state_dir, "loop1")
    assert state.status == "awaiting_plan_approval"
    assert len(channel.asks) == 1
    qid = channel.asks[0]["question_id"]
    engine.tick()
    assert len(channel.asks) == 1  # idempotent

    for phrase in (" OK ", "承認", "進めて", "go"):
        break
    inbox = store._inbox_dir(cfg.state_dir, "loop1")
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "001-a.json").write_text(
        json.dumps(
            {
                "kind": "answer",
                "message_id": "m1",
                "question_id": qid,
                "text": " OK ",
                "received_at": "2026-08-20T12:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )
    engine.tick()
    state = store.load_state(cfg.state_dir, "loop1")
    assert state.status == "executing"
    assert any(e["event"] == "plan_approved" for e in store.read_events(cfg.state_dir, "loop1"))


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_decompose")
def test_plan_approval_false_skips_ask(mock_decompose, mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops()
    mock_decompose.return_value = (
        prompts.DecomposeResponse(
            subtasks=[prompts.DecomposeSubtask(id="s1", title="One", kind="auto", prompt="p")],
            reasoning="",
            uncertain_flag=False,
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel)
    cfg = _config(tmp_path)
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="decomposing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        plan_approval=False,
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg.state_dir, state)
    engine.tick()
    assert store.load_state(cfg.state_dir, "loop1").status == "executing"
    assert channel.asks == []


@patch("mltgnt.loops.engine.load_persona")
@patch("mltgnt.loops.engine.prompts.run_replan")
@patch("mltgnt.loops.engine.prompts.run_decompose")
def test_plan_revision_limit_and_cancel(mock_decompose, mock_replan, mock_persona, tmp_path):
    mock_persona.return_value = _persona_with_ops()
    mock_decompose.return_value = (
        prompts.DecomposeResponse(
            subtasks=[prompts.DecomposeSubtask(id="s1", title="One", kind="auto", prompt="p")],
            reasoning="",
            uncertain_flag=False,
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    mock_replan.return_value = (
        prompts.ReplanResponse(
            keep=(),
            add=(prompts.DecomposeSubtask(id="s1", title="Rev", kind="auto", prompt="p2"),),
            reason="revised",
            reasoning="",
            uncertain_flag=False,
        ),
        prompts.LlmTrace("", "", {}, "", {}, {}, False),
    )
    channel = FakeHumanChannel()
    engine = _engine(tmp_path, channel=channel)
    cfg = _config(tmp_path)
    state = LoopState(
        loop_id="loop1",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="decomposing",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        plan_approval=True,
        created_at="t",
        updated_at="t",
    )
    store.save_state(cfg.state_dir, state)
    engine.tick()
    state = store.load_state(cfg.state_dir, "loop1")
    qid = state.pending_question.question_id
    inbox = store._inbox_dir(cfg.state_dir, "loop1")
    inbox.mkdir(parents=True, exist_ok=True)

    def _answer(text: str, mid: str, q: str):
        (inbox / f"{mid}.json").write_text(
            json.dumps(
                {
                    "kind": "answer",
                    "message_id": mid,
                    "question_id": q,
                    "text": text,
                    "received_at": "2026-08-20T12:00:00+09:00",
                }
            ),
            encoding="utf-8",
        )

    # non-approval → replan (human revision, replan_count untouched)
    _answer("OKですが修正", "m1", qid)
    engine.tick()  # → replanning
    assert store.load_state(cfg.state_dir, "loop1").status == "replanning"
    assert store.load_state(cfg.state_dir, "loop1").replan_count == 0
    engine.tick()  # replan LLM
    state = store.load_state(cfg.state_dir, "loop1")
    assert state.status == "awaiting_plan_approval"
    assert state.plan_revision == 1
    assert state.replan_count == 0

    # burn remaining revisions to hit limit (max_plan_revisions=3)
    for i in range(2, 4):
        q = state.pending_question.question_id
        _answer(f"まだ修正{i}", f"m{i}", q)
        engine.tick()
        engine.tick()
        state = store.load_state(cfg.state_dir, "loop1")
        assert state.plan_revision == i
        assert state.replan_count == 0

    # further non-approval at limit → execute without LLM
    mock_replan.reset_mock()
    q = state.pending_question.question_id
    _answer("まだダメ", "m-limit", q)
    engine.tick()
    state = store.load_state(cfg.state_dir, "loop1")
    assert state.status == "executing"
    mock_replan.assert_not_called()
    assert any("上限" in n["text"] for n in channel.notifies)

    # cancel while awaiting_plan_approval
    state2 = LoopState(
        loop_id="loop2",
        objective_path="/tmp/x.md",
        objective_hash="h",
        title="T",
        body="body",
        status="awaiting_plan_approval",
        iteration=1,
        max_iterations=5,
        persona="mizuho",
        thread=_thread(),
        plan_approval=True,
        pending_question=PendingQuestion(
            question_id="plan-approval-1-0", text="plan", kind="plan_approval"
        ),
        created_at="t",
        updated_at="t",
    )
    cfg2 = _config(tmp_path / "c")
    eng2 = LoopsEngine(
        config=cfg2,
        human_channel=FakeHumanChannel(),
        executor=FakeExecutor(),
        objective_exists=lambda _: True,
        objective_cancelled=lambda _: False,
        objective_hash_changed=lambda *_: False,
    )
    store.save_state(cfg2.state_dir, state2)
    inbox2 = store._inbox_dir(cfg2.state_dir, "loop2")
    inbox2.mkdir(parents=True)
    (inbox2 / "c.json").write_text(
        json.dumps(
            {
                "kind": "cancel",
                "message_id": "c1",
                "question_id": "",
                "text": "",
                "received_at": "2026-08-20T12:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )
    eng2.tick()
    assert store.load_state(cfg2.state_dir, "loop2").status == "cancelled"
