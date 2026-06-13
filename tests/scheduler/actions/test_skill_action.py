"""run_skill_action / _determine_exit_code のユニットテスト（Issue #2076）。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mltgnt.scheduler.actions.skill import _determine_exit_code, run_skill_action
from mltgnt.scheduler.models import ScheduleJob
from mltgnt.skill.models import ExitStatus, SkillMeta

_ENQUEUE = "mltgnt.bridges.ghdag_bridge.enqueue_and_wait"
_ENQUEUE_DAG = "mltgnt.bridges.ghdag_bridge.enqueue_dag"

_FANOUT_RESPONSE = (
    "通常の応答テキスト\n"
    "---\n"
    "ghdag_fanout:\n"
    "  children:\n"
    "    - id: child-1\n"
    "      command: \"agent -p --force < order-1.md\"\n"
    "    - id: child-2\n"
    "      command: \"agent -p --force < order-2.md\"\n"
)


def _make_skill_meta(name: str, tmp_path: Path) -> SkillMeta:
    skill_dir = tmp_path / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"---\nname: {name}\ndescription: test skill\n---\n\nスキル本文",
        encoding="utf-8",
    )
    return SkillMeta(
        name=name,
        description="test skill",
        argument_hint="",
        model=None,
        path=skill_file,
    )


def _make_persona(
    tmp_path: Path,
    name: str = "タチコマ",
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
        "---\n\n## 基本情報\n\nペルソナ本文",
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
            "persona": "タチコマ",
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
        action_args = {"skill": "test-skill", "persona": "タチコマ"}
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


class TestExitStatus:
    def test_constants(self) -> None:
        assert ExitStatus.SUCCESS == 0
        assert ExitStatus.ALREADY_APPLIED == 1
        assert ExitStatus.INVALID_STATE == 2
        assert ExitStatus.USAGE_ERROR == 64


class TestDetermineExitCode:
    def test_success(self) -> None:
        assert _determine_exit_code(True, "応答テキスト") == ExitStatus.SUCCESS

    def test_already_applied(self) -> None:
        msg = "done\nPIPELINE_STATUS: ALREADY_APPLIED"
        assert _determine_exit_code(True, msg) == ExitStatus.ALREADY_APPLIED

    def test_invalid_state(self) -> None:
        msg = "error\nPIPELINE_STATUS: INVALID_STATE"
        assert _determine_exit_code(False, msg) == ExitStatus.INVALID_STATE

    def test_usage_error(self) -> None:
        assert _determine_exit_code(False, "generic error") == ExitStatus.USAGE_ERROR


class TestRunSkillActionExitCodeBranch:
    def test_success_returns_original_msg(self, tmp_path: Path) -> None:
        ok, msg = _run_skill(tmp_path, enqueue_return=(True, "応答テキスト"))
        assert ok is True
        assert msg == "応答テキスト"

    def test_already_applied_returns_idempotent_success(self, tmp_path: Path) -> None:
        ok, msg = _run_skill(
            tmp_path,
            enqueue_return=(True, "PIPELINE_STATUS: ALREADY_APPLIED"),
        )
        assert ok is True
        assert msg == "already_applied"

    def test_invalid_state_returns_retry_false(self, tmp_path: Path) -> None:
        ok, msg = _run_skill(
            tmp_path,
            enqueue_return=(False, "PIPELINE_STATUS: INVALID_STATE"),
        )
        assert ok is False
        assert msg == "invalid_state"

    def test_usage_error_returns_original_msg(self, tmp_path: Path) -> None:
        ok, msg = _run_skill(tmp_path, enqueue_return=(False, "engine error detail"))
        assert ok is False
        assert msg == "engine error detail"

    def test_fanout_bypasses_exit_code_branch(self, tmp_path: Path) -> None:
        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path)
        job = _skill_job(action_args={
            "skill": "test-skill",
            "persona": "タチコマ",
            "enable_fanout": True,
        })

        with patch(_ENQUEUE, return_value=(True, _FANOUT_RESPONSE)), \
             patch(_ENQUEUE_DAG, return_value=[(True, "ok1"), (True, "ok2")]):
            ok, msg = run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert ok is True
        assert "2" in msg and "steps" in msg
