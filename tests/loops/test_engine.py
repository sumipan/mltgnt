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
