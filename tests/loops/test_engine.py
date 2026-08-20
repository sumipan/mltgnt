"""tests/loops/test_engine.py — 状態遷移テスト。"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.config import LoopsConfig
from mltgnt.loops.engine import LoopsEngine
from mltgnt.loops.models import LoopState, PendingQuestion, Subtask
from mltgnt.loops.objective import Objective
from mltgnt.loops import store
from mltgnt.loops import prompts
from mltgnt.interfaces.loops import StepPoll, StepSubmission
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


def _objective(loop_id: str = "loop1") -> Objective:
    return Objective(
        loop_id=loop_id,
        title="Title",
        body="Do something",
        agent="mizuho",
        max_iterations=5,
        status="active",
        path=Path(f"/tmp/{loop_id}.md"),
        content_hash="hash1",
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
