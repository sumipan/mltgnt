"""spawn_job posting and post resolver tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

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


def _make_slack_mock() -> MagicMock:
    slack = MagicMock()
    slack.post_message = MagicMock()
    return slack


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
