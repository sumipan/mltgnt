"""tests/scheduler/actions/test_dream_action.py — memory_dream action tests."""
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
    """Stand-in for ghdag.llm.TextResult (only body is read)."""
    return type("R", (), {"body": body, "success": True, "stderr": "", "returncode": 0})()


def test_run_dream_action_success(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    persona_dir = _setup_persona_with_jsonl(agents_dir, "alice")
    config = _memory_config(tmp_path)
    job = _dream_job()

    llm_response = _text_result("## Behavior patterns\nmorning person\n\n## Preferences\nconcise")

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=llm_response):
        ok, msg = run_dream_action(job, persona_dir=persona_dir, memory_config=config)

    assert ok is True
    assert "synthesized" in msg
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
    llm_response = _text_result("## Behavior patterns\npattern\n\n## Preferences\npref")

    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=llm_response):
        ok, msg = sch.execute_action(job)

    assert ok is True
    assert "synthesized" in msg


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


def _run_with_config(tmp_path: Path, config: MemoryConfig):
    persona_dir = _setup_persona_with_jsonl(tmp_path / "agents", "alice")
    llm_response = _text_result("## Behavior patterns\npattern\n\n## Preferences\npref")
    with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=llm_response) as mock_call:
        ok, _msg = run_dream_action(_dream_job(), persona_dir=persona_dir, memory_config=config)
    assert ok is True
    return mock_call.call_args.kwargs


@pytest.mark.parametrize(
    ("overrides", "expected_engine", "expected_model"),
    [
        ({}, "claude", "claude-haiku-4-5-20251001"),
        ({"dream_engine": ""}, "claude", "claude-haiku-4-5-20251001"),
        # Empty model is forwarded as None so ghdag picks the engine default
        # (ghdag rejects "" in its model allowlist check).
        ({"dream_engine": "cursor"}, "cursor", None),
        ({"dream_engine": "codex"}, "codex", None),
        ({"dream_engine": "cursor", "dream_model": "X"}, "cursor", "X"),
    ],
)
def test_run_dream_action_passes_resolved_engine_and_model(
    tmp_path: Path,
    overrides: dict[str, str],
    expected_engine: str,
    expected_model: str | None,
) -> None:
    config = MemoryConfig(chat_dir=tmp_path, use_dream_summary=True, **overrides)
    kwargs = _run_with_config(tmp_path, config)
    assert kwargs["engine"] == expected_engine
    assert kwargs["model"] == expected_model


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, ("claude", "claude-haiku-4-5-20251001")),
        ({"dream_engine": ""}, ("claude", "claude-haiku-4-5-20251001")),
        ({"dream_engine": " cursor "}, ("cursor", "")),
        ({"dream_engine": "codex"}, ("codex", "")),
        ({"dream_engine": "codex", "dream_model": "X"}, ("codex", "X")),
        ({"dream_model": "custom"}, ("claude", "custom")),
    ],
)
def test_resolve_dream_llm(
    tmp_path: Path, overrides: dict[str, str], expected: tuple[str, str]
) -> None:
    from mltgnt.scheduler.actions.dream import _resolve_dream_llm

    config = MemoryConfig(chat_dir=tmp_path, **overrides)
    assert _resolve_dream_llm(config) == expected
