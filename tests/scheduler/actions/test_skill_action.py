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


class TestEnablePipeline:
    """Issue #3031: enable_pipeline 経路。"""

    def test_enable_pipeline_calls_compose_and_enqueue_dag(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock

        from mltgnt.skill.models import ConsumesSpec, ProducesSpec, SkillMatchResult

        persona_dir = _make_persona(tmp_path)
        meta_a = _make_skill_meta("skill-a", tmp_path)
        meta_a.skill_io = "v1"
        meta_a.produces = ProducesSpec(content_type="text/markdown")
        meta_b = _make_skill_meta("skill-b", tmp_path)
        meta_b.skill_io = "v1"
        meta_b.produces = ProducesSpec(content_type="text/markdown")
        meta_b.consumes = [
            ConsumesSpec(producer="skill-a", content_type="text/markdown")
        ]
        job = _skill_job(
            action_args={
                "skill": "skill-a",
                "persona": "タチコマ",
                "argv": ["/skill-a", "foo", "|", "/skill-b"],
                "enable_pipeline": True,
            }
        )
        match_results = [
            SkillMatchResult(
                decisive=meta_a,
                candidates=[meta_a],
                rationale="slash:skill-a",
                arguments="foo",
            ),
            SkillMatchResult(
                decisive=meta_b,
                candidates=[meta_b],
                rationale="slash:skill-b",
                arguments="",
            ),
        ]
        captured: dict = {}

        def capture_dag(steps, **kwargs):
            captured["steps"] = list(steps)
            return [(True, "out-a"), (True, "out-b")]

        with (
            patch(
                "mltgnt.skill.matcher.match_pipeline",
                new_callable=AsyncMock,
                return_value=match_results,
            ),
            patch(_ENQUEUE_DAG, side_effect=capture_dag),
            patch(_ENQUEUE) as mock_single,
        ):
            ok, msg = run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"skill-a": meta_a, "skill-b": meta_b},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert ok is True
        assert msg == "out-b"
        assert len(captured["steps"]) == 2
        assert captured["steps"][0].id == "pipe_0_skill-a"
        assert captured["steps"][1].id == "pipe_1_skill-b"
        assert captured["steps"][1].depends == ["pipe_0_skill-a"]
        mock_single.assert_not_called()

    def test_enable_pipeline_prefers_over_fanout(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock

        from mltgnt.skill.models import SkillMatchResult

        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path)
        job = _skill_job(
            action_args={
                "skill": "test-skill",
                "persona": "タチコマ",
                "argv": ["/test-skill", "x"],
                "enable_pipeline": True,
                "enable_fanout": True,
            }
        )
        match_results = [
            SkillMatchResult(
                decisive=meta,
                candidates=[meta],
                rationale="slash:test-skill",
                arguments="x",
            )
        ]

        with (
            patch(_ENQUEUE) as mock_single,
            patch(
                "mltgnt.skill.matcher.match_pipeline",
                new_callable=AsyncMock,
                return_value=match_results,
            ),
            patch(_ENQUEUE_DAG, return_value=[(True, "pipe-done")]) as mock_dag,
        ):
            ok, msg = run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert ok is True
        assert msg == "pipe-done"
        mock_single.assert_not_called()
        mock_dag.assert_called_once()


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


def _write_knowledge(skill_meta: SkillMeta, text: str) -> None:
    path = skill_meta.path.parent / "knowledge.md"
    path.write_text(text, encoding="utf-8")
    skill_meta.knowledge_paths = [path]


def _write_memory(repo_root: Path, persona_name: str, text: str) -> None:
    mem_dir = repo_root / "chat" / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / f"{persona_name}.jsonl").write_text(text, encoding="utf-8")


def _context_injection_records(audit_path: Path) -> list[dict]:
    if not audit_path.exists():
        return []
    return [
        json.loads(line)
        for line in audit_path.read_text().splitlines()
        if line.strip() and json.loads(line).get("event_type") == "context_injection"
    ]


class TestContextInjection:
    """Issue #3021: knowledge.md × 記憶ファイルの 4 組み合わせと audit。"""

    def _capture_prompt(
        self,
        tmp_path: Path,
        *,
        with_knowledge: bool,
        with_memory: bool,
        knowledge_count: int = 5,
        memory_max_bytes: int = 4096,
    ) -> tuple[str, list[dict]]:
        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path)
        if with_knowledge:
            _write_knowledge(
                meta,
                "知1\n\n知2\n\n知3\n\n知4\n\n知5\n\n知6",
            )
        if with_memory:
            _write_memory(tmp_path, "タチコマ", '{"ts":"2026-09-09","text":"昨夜の話"}\n')
        (tmp_path / "jobs").mkdir(exist_ok=True)
        job = _skill_job(
            action_args={
                "skill": "test-skill",
                "persona": "タチコマ",
                "knowledge_count": knowledge_count,
                "memory_max_bytes": memory_max_bytes,
            }
        )
        captured: dict = {}

        def capture_enqueue(**kwargs):
            captured["prompt"] = kwargs["prompt"]
            return True, "ok"

        with patch(_ENQUEUE, side_effect=capture_enqueue):
            ok, _ = run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )
        assert ok is True
        records = _context_injection_records(tmp_path / "jobs" / "audit.jsonl")
        return captured["prompt"], records

    def test_neither_knowledge_nor_memory(self, tmp_path: Path) -> None:
        prompt, records = self._capture_prompt(
            tmp_path, with_knowledge=False, with_memory=False
        )
        assert "## コンテキスト" not in prompt
        assert len(records) == 1
        assert records[0]["knowledge_count"] == 0
        assert records[0]["memory_bytes"] == 0

    def test_knowledge_only(self, tmp_path: Path) -> None:
        prompt, records = self._capture_prompt(
            tmp_path, with_knowledge=True, with_memory=False, knowledge_count=3
        )
        assert "## コンテキスト" in prompt
        assert "### knowledge（直近 3 件）" in prompt
        assert "知4" in prompt and "知5" in prompt and "知6" in prompt
        assert "知3" not in prompt
        assert "### 記憶（末尾）" not in prompt
        assert records[0]["knowledge_count"] == 3
        assert records[0]["memory_bytes"] == 0

    def test_memory_only(self, tmp_path: Path) -> None:
        prompt, records = self._capture_prompt(
            tmp_path, with_knowledge=False, with_memory=True
        )
        assert "## コンテキスト" in prompt
        assert "### knowledge" not in prompt
        assert "### 記憶（末尾）" in prompt
        assert "昨夜の話" in prompt
        assert records[0]["knowledge_count"] == 0
        assert records[0]["memory_bytes"] > 0

    def test_both_knowledge_and_memory(self, tmp_path: Path) -> None:
        prompt, records = self._capture_prompt(
            tmp_path, with_knowledge=True, with_memory=True, knowledge_count=2
        )
        assert "## コンテキスト" in prompt
        assert "### knowledge（直近 2 件）" in prompt
        assert "### 記憶（末尾）" in prompt
        assert "知5" in prompt and "知6" in prompt
        assert "昨夜の話" in prompt
        assert records[0]["event_type"] == "context_injection"
        assert records[0]["skill_name"] == "test-skill"
        assert records[0]["job_id"] == "skill_job"
        assert records[0]["schema_version"] == 1
        assert records[0]["knowledge_count"] == 2
        assert records[0]["memory_bytes"] > 0
        assert "timestamp" in records[0]
