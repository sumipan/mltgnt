"""Shared helpers for skill action scheduler tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mltgnt.scheduler.actions.skill import run_skill_action
from mltgnt.scheduler.models import ScheduleJob
from mltgnt.skill.models import ProducesSpec, SideEffectsSpec, SkillMeta

_ENQUEUE = "mltgnt.bridges.ghdag_bridge.enqueue_and_wait"
_ENQUEUE_DAG = "mltgnt.bridges.ghdag_bridge.enqueue_dag"

_FANOUT_RESPONSE = (
    "normal response text\n"
    "---\n"
    "ghdag_fanout:\n"
    "  children:\n"
    "    - id: child-1\n"
    "      command: \"agent -p --force < order-1.md\"\n"
    "    - id: child-2\n"
    "      command: \"agent -p --force < order-2.md\"\n"
)


def _make_skill_meta(
    name: str,
    tmp_path: Path,
    side_effects: SideEffectsSpec | None = None,
    *,
    status_markers: list[str] | None = None,
) -> SkillMeta:
    skill_dir = tmp_path / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    fm_lines = [f"name: {name}", "description: test skill"]
    produces: ProducesSpec | None = None
    if status_markers is not None:
        fm_lines.append("skill_io: v1")
        fm_lines.append("produces:")
        fm_lines.append("  content_type: text/plain")
        fm_lines.append("  status_markers:")
        for marker in status_markers:
            fm_lines.append(f"    - {marker}")
        produces = ProducesSpec(
            content_type="text/plain", status_markers=list(status_markers)
        )
    skill_file.write_text(
        "---\n" + "\n".join(fm_lines) + "\n---\n\nskill body",
        encoding="utf-8",
    )
    return SkillMeta(
        name=name,
        description="test skill",
        argument_hint="",
        model=None,
        path=skill_file,
        side_effects=side_effects,
        skill_io="v1" if status_markers is not None else "legacy",
        produces=produces,
    )


def _make_persona(
    tmp_path: Path,
    name: str = "persona-a",
    engine: str = "claude",
    model: str = "claude-sonnet-4-6",
) -> Path:
    persona_dir = tmp_path / "agents"
    persona_dir.mkdir(parents=True, exist_ok=True)
    p = persona_dir / f"{name}.md"
    p.write_text(
        "---\n"
        f"persona:\n  name: {name}\n"
        f"ops:\n  engine: {engine}\n  model: {model}\n"
        "---\n\n## Basic info\n\npersona body",
        encoding="utf-8",
    )
    return persona_dir


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


def _run_skill(
    tmp_path: Path,
    *,
    enqueue_return: tuple[bool, str],
    job: ScheduleJob | None = None,
    enable_fanout: bool = False,
) -> tuple[bool, str]:
    persona_dir = _make_persona(tmp_path)
    meta = _make_skill_meta("test-skill", tmp_path)
    if job is None:
        action_args = {"skill": "test-skill", "persona": "persona-a"}
        if enable_fanout:
            action_args["enable_fanout"] = True
        job = _skill_job(action_args=action_args)

    with patch(_ENQUEUE, return_value=enqueue_return):
        return run_skill_action(
            job,
            persona_dir=persona_dir,
            skill_registry={"test-skill": meta},
            default_tz="Asia/Tokyo",
            repo_root=tmp_path,
        )
