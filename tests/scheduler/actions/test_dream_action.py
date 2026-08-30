"""tests/scheduler/actions/test_dream_action.py — memory_dream アクションのテスト。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mltgnt.config import MemoryConfig
from mltgnt.memory.dream import read_dream
from mltgnt.scheduler import PersonaScheduler, ScheduleJob
from mltgnt.scheduler.actions.dream import run_dream_action


def _memory_config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(
        chat_dir=tmp_path,
        use_dream_summary=True,
        dream_model="test-model",
    )


def _dream_job(persona: str = "alice") -> ScheduleJob:
    return ScheduleJob.from_dict({
        "id": "dream_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "memory_dream",
        "notify": "silent",
        "persona": persona,
    })


def _setup_persona_with_jsonl(agents_dir: Path, persona: str) -> Path:
    persona_dir = agents_dir / persona
    memory_dir = persona_dir / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "chat.jsonl").write_text(
        '{"timestamp":"2026-06-01 10:00","role":"user","content":"hello","source_tag":"chat"}\n',
        encoding="utf-8",
    )
    return persona_dir


def _text_result(body: str):
    """ghdag.llm.TextResult 相当（body のみ参照される）。"""
    return type("R", (), {"body": body, "success": True, "stderr": "", "returncode": 0})()


def test_run_dream_action_success(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    persona_dir = _setup_persona_with_jsonl(agents_dir, "alice")
    config = _memory_config(tmp_path)
    job = _dream_job()

    llm_response = _text_result("## 行動パターン\n朝型\n\n## 好み・傾向\n簡潔")

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=llm_response):
        ok, msg = run_dream_action(job, persona_dir=persona_dir, memory_config=config)

    assert ok is True
    assert "合成しました" in msg
    loaded = read_dream(persona_dir)
    assert loaded is not None
    assert loaded.persona == "alice"
    assert len(loaded.sections) == 2


def test_run_dream_action_llm_failure_returns_false(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    persona_dir = _setup_persona_with_jsonl(agents_dir, "alice")
    config = _memory_config(tmp_path)
    job = _dream_job()

    with patch("mltgnt.bridges.llm_adapter.call_llm", side_effect=RuntimeError("boom")):
        ok, msg = run_dream_action(job, persona_dir=persona_dir, memory_config=config)

    assert ok is False
    assert "boom" in msg


def test_memory_dream_not_registered_when_disabled(tmp_path: Path) -> None:
    config = MemoryConfig(chat_dir=tmp_path, use_dream_summary=False)
    sch = PersonaScheduler(
        slack=None,
        state_dir=tmp_path / "state",
        jobs=[],
        memory_config=config,
    )
    job = _dream_job()
    with pytest.raises(ValueError):
        sch.execute_action(job)


def test_memory_dream_registered_and_fires(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    _setup_persona_with_jsonl(agents_dir, "alice")
    config = _memory_config(tmp_path)
    sch = PersonaScheduler(
        slack=None,
        state_dir=tmp_path / "state",
        jobs=[],
        persona_dir=agents_dir,
        memory_config=config,
    )
    job = _dream_job()
    llm_response = _text_result("## 行動パターン\npattern\n\n## 好み・傾向\npref")

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=llm_response):
        ok, msg = sch.execute_action(job)

    assert ok is True
    assert "合成しました" in msg


def test_persona_scheduler_default_memory_config_unchanged(tmp_path: Path) -> None:
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[])
    job = ScheduleJob.from_dict({
        "id": "noop_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    ok, msg = sch.execute_action(job)
    assert ok is True
    assert msg == ""
