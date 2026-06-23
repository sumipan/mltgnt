"""run_skill_action / _determine_exit_code のユニットテスト（Issue #2076）。"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mltgnt.scheduler.actions.skill import (
    _compute_write_diff,
    _determine_exit_code,
    _snapshot_writes,
    run_skill_action,
)
from mltgnt.scheduler.models import ScheduleJob
from mltgnt.skill.models import ExitStatus, SkillMeta, SideEffectsSpec

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


def _make_skill_meta(
    name: str, tmp_path: Path, side_effects: SideEffectsSpec | None = None
) -> SkillMeta:
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
        side_effects=side_effects,
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


class TestRunSkillActionPermissionPassthrough:
    def test_permission_none_when_not_in_action_args(self, tmp_path: Path) -> None:
        """action_args に permission キーがない場合、enqueue_and_wait に permission=None が渡される。"""
        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path)
        job = _skill_job(action_args={"skill": "test-skill", "persona": "タチコマ"})
        captured_kwargs: dict = {}

        def capture_enqueue(**kwargs):
            captured_kwargs.update(kwargs)
            return True, "ok"

        with patch(_ENQUEUE, side_effect=capture_enqueue):
            run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert captured_kwargs.get("permission") is None

    def test_permission_passed_from_action_args(self, tmp_path: Path) -> None:
        """action_args.permission='dangerous_full_access' が enqueue_and_wait に渡される。"""
        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path)
        job = _skill_job(action_args={
            "skill": "test-skill",
            "persona": "タチコマ",
            "permission": "dangerous_full_access",
        })
        captured_kwargs: dict = {}

        def capture_enqueue(**kwargs):
            captured_kwargs.update(kwargs)
            return True, "ok"

        with patch(_ENQUEUE, side_effect=capture_enqueue):
            run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert captured_kwargs.get("permission") == "dangerous_full_access"


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


class TestSnapshotWrites:
    def test_empty_patterns_returns_empty(self, tmp_path: Path) -> None:
        assert _snapshot_writes([], tmp_path) == {}

    def test_captures_matching_file(self, tmp_path: Path) -> None:
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        f = jobs_dir / "audit.jsonl"
        f.write_text("x")
        result = _snapshot_writes(["jobs/*.jsonl"], tmp_path)
        assert "jobs/audit.jsonl" in result

    def test_no_match_returns_empty(self, tmp_path: Path) -> None:
        result = _snapshot_writes(["jobs/*.jsonl"], tmp_path)
        assert result == {}


class TestComputeWriteDiff:
    def test_new_file_detected(self) -> None:
        before: dict[str, float] = {}
        after = {"jobs/audit.jsonl": 1.0}
        assert _compute_write_diff(before, after) == ["jobs/audit.jsonl"]

    def test_modified_file_detected(self) -> None:
        before = {"jobs/audit.jsonl": 1.0}
        after = {"jobs/audit.jsonl": 2.0}
        assert _compute_write_diff(before, after) == ["jobs/audit.jsonl"]

    def test_unchanged_file_not_included(self) -> None:
        before = {"jobs/audit.jsonl": 1.0}
        after = {"jobs/audit.jsonl": 1.0}
        assert _compute_write_diff(before, after) == []

    def test_deleted_file_not_included(self) -> None:
        before = {"jobs/audit.jsonl": 1.0}
        after: dict[str, float] = {}
        assert _compute_write_diff(before, after) == []


class TestSideEffectAuditIntegration:
    def _run_with_side_effects(
        self,
        tmp_path: Path,
        *,
        side_effects: SideEffectsSpec | None,
        enqueue_return: tuple[bool, str] = (True, "ok"),
        write_file: bool = False,
    ) -> tuple[bool, str]:
        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path, side_effects=side_effects)
        job = _skill_job(action_args={"skill": "test-skill", "persona": "タチコマ"})
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir(exist_ok=True)

        def fake_enqueue(**kwargs):
            if write_file:
                (jobs_dir / "audit.jsonl").write_text("existing\n")
            return enqueue_return

        with patch(_ENQUEUE, side_effect=fake_enqueue):
            return run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

    def test_audit_record_written_when_writes_declared(self, tmp_path: Path) -> None:
        se = SideEffectsSpec(writes=["jobs/*.jsonl"])
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        audit_path = jobs_dir / "audit.jsonl"

        meta = _make_skill_meta("test-skill", tmp_path, side_effects=se)
        job = _skill_job(action_args={"skill": "test-skill", "persona": "タチコマ"})
        persona_dir = _make_persona(tmp_path)

        def fake_enqueue(**kwargs):
            audit_path.write_text(json.dumps({"event_type": "other"}) + "\n")
            return (True, "ok")

        with patch(_ENQUEUE, side_effect=fake_enqueue):
            ok, _ = run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert ok is True
        lines = audit_path.read_text().splitlines()
        audit_records = [
            json.loads(line)
            for line in lines
            if line.strip() and json.loads(line).get("event_type") == "side_effect_audit"
        ]
        assert len(audit_records) == 1
        rec = audit_records[0]
        assert rec["skill_name"] == "test-skill"
        assert rec["declared_writes"] == ["jobs/*.jsonl"]
        assert "jobs/audit.jsonl" in rec["actual_writes"]
        assert rec["schema_version"] == 1

    def test_no_audit_when_side_effects_is_none(self, tmp_path: Path) -> None:
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        audit_path = jobs_dir / "audit.jsonl"

        self._run_with_side_effects(tmp_path, side_effects=None)

        if audit_path.exists():
            lines = [
                json.loads(line)
                for line in audit_path.read_text().splitlines()
                if line.strip()
            ]
            assert not any(line.get("event_type") == "side_effect_audit" for line in lines)

    def test_no_audit_when_writes_is_empty_list(self, tmp_path: Path) -> None:
        se = SideEffectsSpec(writes=[])
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        audit_path = jobs_dir / "audit.jsonl"

        self._run_with_side_effects(tmp_path, side_effects=se)

        if audit_path.exists():
            lines = [
                json.loads(line)
                for line in audit_path.read_text().splitlines()
                if line.strip()
            ]
            assert not any(line.get("event_type") == "side_effect_audit" for line in lines)

    def test_all_declared_covered_false_when_unmatched(self, tmp_path: Path) -> None:
        se = SideEffectsSpec(writes=["jobs/*.md"])
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        audit_path = jobs_dir / "audit.jsonl"

        meta = _make_skill_meta("test-skill", tmp_path, side_effects=se)
        job = _skill_job(action_args={"skill": "test-skill", "persona": "タチコマ"})
        persona_dir = _make_persona(tmp_path)

        def fake_enqueue(**kwargs):
            (jobs_dir / "audit.jsonl").write_text(json.dumps({"event_type": "other"}) + "\n")
            return (True, "ok")

        with patch(_ENQUEUE, side_effect=fake_enqueue):
            run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        lines = audit_path.read_text().splitlines()
        records = [
            json.loads(line)
            for line in lines
            if line.strip() and json.loads(line).get("event_type") == "side_effect_audit"
        ]
        assert len(records) == 1
        assert records[0]["all_declared_covered"] is False

    def test_oserror_does_not_fail_skill(self, tmp_path: Path) -> None:
        se = SideEffectsSpec(writes=["jobs/*.jsonl"])
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        audit_path = jobs_dir / "audit.jsonl"
        audit_path.write_text("")
        audit_path.chmod(0o444)

        meta = _make_skill_meta("test-skill", tmp_path, side_effects=se)
        job = _skill_job(action_args={"skill": "test-skill", "persona": "タチコマ"})
        persona_dir = _make_persona(tmp_path)

        with patch(_ENQUEUE, return_value=(True, "ok")):
            ok, _ = run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert ok is True
        audit_path.chmod(0o644)
