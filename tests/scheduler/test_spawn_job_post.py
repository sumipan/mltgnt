"""spawn_job posting and post resolver tests."""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.scheduler import PersonaScheduler, ScheduleJob
from mltgnt.skill.models import SkillMeta


def _make_skill_meta(name: str, tmp_path: Path) -> SkillMeta:
    skill_dir = tmp_path / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: {}\ndescription: test skill\n---\n\nskill body".format(name),
        encoding="utf-8",
    )
    return SkillMeta(
        name=name,
        description="test skill",
        argument_hint="",
        model=None,
        path=skill_file,
    )


def _make_persona(tmp_path: Path, name: str, engine: str = "claude", model: str = "claude-sonnet-4-6") -> Path:
    persona_dir = tmp_path / "agents"
    persona_dir.mkdir(parents=True, exist_ok=True)
    p = persona_dir / f"{name}.md"
    p.write_text(
        "---\n"
        f"persona:\n  name: {name}\n"
        f"ops:\n  engine: {engine}\n  model: {model}\n"
        "---\n\n## Basic information\n\npersona body",
        encoding="utf-8",
    )
    return p


def _make_skill_scheduler(tmp_path: Path, skill_name: str = "test-skill") -> tuple[PersonaScheduler, SkillMeta]:
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[], repo_root=tmp_path)
    meta = _make_skill_meta(skill_name, tmp_path)
    sch._skill_registry = {skill_name: meta}
    return sch, meta


def _skill_job(**overrides) -> ScheduleJob:
    defaults = dict(
        id="skill_job",
        mode="scheduled",
        action="skill",
        notify="silent",
        every_day_at="10:00",
        action_args={
            "skill": "test-skill",
            "persona": "persona-a",
        },
    )
    defaults.update(overrides)
    return ScheduleJob.from_dict(defaults)


_ENQUEUE = "mltgnt.bridges.ghdag_bridge.enqueue_and_wait"

def _enqueue_side_effect(return_value=(True, "")):
    captured: list[dict] = []

    def _fake(**kwargs):
        captured.append(kwargs)
        return return_value

    return captured, _fake


def _dag_side_effect(return_value=None):
    if return_value is None:
        return_value = [(True, "ok")]
    captured: list[dict] = []
    positional: list = []

    def _fake(*args, **kwargs):
        if args:
            positional.append(args[0])
        captured.append(kwargs)
        return return_value

    return captured, positional, _fake
def _make_slack_mock() -> MagicMock:
    slack = MagicMock()
    slack.post_message = MagicMock()
    return slack


def _command_job(**overrides) -> ScheduleJob:
    defaults = dict(
        id="command_job",
        mode="scheduled",
        action="command",
        notify="slack_secretary",
        every_day_at="10:00",
        action_args={"command": "echo hello"},
    )
    defaults.update(overrides)
    return ScheduleJob.from_dict(defaults)


def test_spawn_job_skill_success_posts_message(tmp_path: Path) -> None:
    """Skill success with notify posts the enqueue result via _post."""
    slack = _make_slack_mock()
    job = _skill_job(notify="slack_secretary")
    sch = PersonaScheduler(slack=slack, state_dir=tmp_path / "state", jobs=[job], repo_root=tmp_path)
    meta = _make_skill_meta("test-skill", tmp_path)
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    sch.reload_jobs()

    post_calls: list[tuple[ScheduleJob, str]] = []

    def _capture_post(j: ScheduleJob, text: str) -> None:
        post_calls.append((j, text))

    with patch(_ENQUEUE, return_value=(True, "hello")):
        with patch.object(sch, "_post", side_effect=_capture_post):
            sch._spawn_job(job, date(2026, 4, 23))
            time.sleep(0.5)

    assert post_calls[-1] == (job, "hello")


def test_spawn_job_skill_empty_result_skips_post(tmp_path: Path) -> None:
    """Skill success with empty message does not call _post."""
    slack = _make_slack_mock()
    job = _skill_job(notify="slack_secretary")
    sch = PersonaScheduler(slack=slack, state_dir=tmp_path / "state", jobs=[job], repo_root=tmp_path)
    meta = _make_skill_meta("test-skill", tmp_path)
    sch._skill_registry = {"test-skill": meta}
    _make_persona(tmp_path, "persona-a")
    sch.reload_jobs()

    post_calls: list[tuple[ScheduleJob, str]] = []

    def _capture_post(j: ScheduleJob, text: str) -> None:
        post_calls.append((j, text))

    with patch(_ENQUEUE, return_value=(True, "")):
        with patch.object(sch, "_post", side_effect=_capture_post):
            sch._spawn_job(job, date(2026, 4, 23))
            time.sleep(0.5)

    assert post_calls == []


def test_spawn_job_command_success_posts_stdout(tmp_path: Path) -> None:
    """Command success with output posts via _post."""
    slack = _make_slack_mock()
    job = _command_job(notify="slack_secretary")
    sch = PersonaScheduler(slack=slack, state_dir=tmp_path / "state", jobs=[job], repo_root=tmp_path)
    sch.reload_jobs()

    post_calls: list[tuple[ScheduleJob, str]] = []

    def _capture_post(j: ScheduleJob, text: str) -> None:
        post_calls.append((j, text))

    with patch.object(sch, "execute_action", return_value=(True, "stdout output")):
        with patch.object(sch, "_post", side_effect=_capture_post):
            sch._spawn_job(job, date(2026, 4, 23))
            time.sleep(0.5)

    assert post_calls[-1] == (job, "stdout output")


def test_post_resolver_does_not_overwrite_skill_text(tmp_path: Path) -> None:
    """AC-4: _post() Home resolver Home text does not override the skill generation text."""
    slack = _make_slack_mock()
    job = _skill_job(notify="slack_secretary", persona="persona-a")

    def resolver(persona_name: str, repo_root: Path) -> tuple[dict, str]:
        return {"icon_emoji": ":robot:"}, "resolver Text"

    sch = PersonaScheduler(
        slack=slack,
        state_dir=tmp_path / "state",
        jobs=[],
        repo_root=tmp_path,
        persona_post_kwargs_resolver=resolver,
    )
    sch.reload_jobs()

    posts: list[tuple[str, dict]] = []

    def _record_post(text: str, channel: str, **kwargs) -> bool:
        posts.append((text, kwargs))
        return True

    slack.post_message.side_effect = _record_post
    sch._post(job, "skill generation text")

    assert posts[-1][0] == "skill generation text"
    assert posts[-1][1].get("icon_emoji") == ":robot:"


def test_post_empty_text_uses_resolver_fallback(tmp_path: Path) -> None:
    """AC-5: text empty resolver Home text Use fallback when applying."""
    slack = _make_slack_mock()
    job = _skill_job(notify="slack_secretary", persona="persona-a")

    def resolver(persona_name: str, repo_root: Path) -> tuple[dict, str]:
        return {"icon_emoji": ":robot:"}, "fallback Text"

    sch = PersonaScheduler(
        slack=slack,
        state_dir=tmp_path / "state",
        jobs=[],
        repo_root=tmp_path,
        persona_post_kwargs_resolver=resolver,
    )
    sch.reload_jobs()

    posts: list[tuple[str, dict]] = []

    def _record_post(text: str, channel: str, **kwargs) -> bool:
        posts.append((text, kwargs))
        return True

    slack.post_message.side_effect = _record_post
    sch._post(job, "")

    assert posts[-1][0] == "fallback Text"


def test_post_resolver_exception_uses_default_kwargs(tmp_path: Path) -> None:
    """AC-6: resolver when an exception is sent default_slack_post_kwargs Use text not changed."""
    slack = _make_slack_mock()
    job = _skill_job(notify="slack_secretary", persona="persona-a")

    def resolver(persona_name: str, repo_root: Path) -> tuple[dict, str]:
        raise RuntimeError("resolver error")

    def default_kwargs() -> dict:
        return {"icon_emoji": ":default:"}

    sch = PersonaScheduler(
        slack=slack,
        state_dir=tmp_path / "state",
        jobs=[],
        repo_root=tmp_path,
        persona_post_kwargs_resolver=resolver,
        default_slack_post_kwargs=default_kwargs,
    )
    sch.reload_jobs()

    posts: list[tuple[str, dict]] = []

    def _record_post(text: str, channel: str, **kwargs) -> bool:
        posts.append((text, kwargs))
        return True

    slack.post_message.side_effect = _record_post
    sch._post(job, "Original text")

    assert posts[-1][0] == "Original text"
    assert posts[-1][1].get("icon_emoji") == ":default:"
