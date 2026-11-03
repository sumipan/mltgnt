"""Side-effect snapshot and audit tests for run_skill_action."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mltgnt.scheduler.actions.skill import (
    _compute_write_diff,
    _snapshot_writes,
    run_skill_action,
)
from mltgnt.skill.models import SideEffectsSpec

from .skill_helpers import (
    _ENQUEUE,
    _ENQUEUE_DAG,
    _FANOUT_RESPONSE,
    _make_persona,
    _make_skill_meta,
    _skill_job,
)

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
        job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a"})
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
        job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a"})
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
        job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a"})
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
        job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a"})
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
