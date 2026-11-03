"""Unit tests for run_skill_action / _determine_exit_code (Issue #2076)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mltgnt.scheduler.actions.skill import (
    _compute_write_diff,
    _determine_exit_code,
    _extract_status_marker,
    _matches_expected,
    _resolve_exit_code,
    _snapshot_writes,
    run_skill_action,
)
from mltgnt.scheduler.models import ScheduleJob
from mltgnt.skill.models import ExitStatus, ProducesSpec, SkillMeta, SideEffectsSpec

from .skill_helpers import (
    _ENQUEUE,
    _ENQUEUE_DAG,
    _FANOUT_RESPONSE,
    _make_persona,
    _make_skill_meta,
    _run_skill,
    _skill_job,
)

class TestExitStatus:
    def test_constants(self) -> None:
        assert ExitStatus.SUCCESS == 0
        assert ExitStatus.ALREADY_APPLIED == 1
        assert ExitStatus.INVALID_STATE == 2
        assert ExitStatus.CONTRACT_VIOLATION == 3
        assert ExitStatus.USAGE_ERROR == 64


class TestDetermineExitCode:
    def test_success(self) -> None:
        assert _determine_exit_code(True, "response text") == ExitStatus.SUCCESS

    def test_already_applied(self) -> None:
        msg = "done\nPIPELINE_STATUS: ALREADY_APPLIED"
        assert _determine_exit_code(True, msg) == ExitStatus.ALREADY_APPLIED

    def test_invalid_state(self) -> None:
        msg = "error\nPIPELINE_STATUS: INVALID_STATE"
        assert _determine_exit_code(False, msg) == ExitStatus.INVALID_STATE

    def test_usage_error(self) -> None:
        assert _determine_exit_code(False, "generic error") == ExitStatus.USAGE_ERROR


_NEXUS_MARKERS = ["ACCEPTED", "ACCEPTED:NO_ACTION", "REJECTED:missing_args"]
_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "marker_response"


class TestResolveExitCode:
    """Issue #3040 / #3172: expected_markers matching and diagnostics."""

    def test_marker_matches_expected(self) -> None:
        code, diagnostics = _resolve_exit_code(
            True, "PIPELINE_STATUS: DONE\ndetail text", ["DONE"]
        )
        assert code == ExitStatus.SUCCESS
        assert isinstance(diagnostics, list)

    def test_marker_undeclared(self) -> None:
        # #3172: with enforce=False (default), exit follows _determine_exit_code
        code, diagnostics = _resolve_exit_code(
            True, "PIPELINE_STATUS: BRUSHUP_DONE\n...", ["DONE"]
        )
        assert code == ExitStatus.SUCCESS
        assert "contract_violation: undeclared PIPELINE_STATUS marker" in diagnostics
        assert "violated_marker=BRUSHUP_DONE" in diagnostics

    def test_marker_missing_from_output(self) -> None:
        # #3172: with enforce=False (default), missing marker does not become CONTRACT_VIOLATION
        code, diagnostics = _resolve_exit_code(True, "processed", ["DONE"])
        assert code == ExitStatus.SUCCESS
        assert "contract_violation: PIPELINE_STATUS marker missing" in diagnostics
        assert "marker=<absent>" in diagnostics

    def test_empty_expected_fallback(self) -> None:
        code, diagnostics = _resolve_exit_code(
            True, "PIPELINE_STATUS: ALREADY_APPLIED", []
        )
        assert code == ExitStatus.ALREADY_APPLIED
        assert isinstance(diagnostics, list)

    def test_bare_marker_accepted_no_action(self) -> None:
        code, diagnostics = _resolve_exit_code(
            True, "ACCEPTED:NO_ACTION\n", _NEXUS_MARKERS
        )
        assert code == ExitStatus.SUCCESS
        assert "marker=ACCEPTED:NO_ACTION" in diagnostics

    def test_bare_marker_accepted_with_note(self) -> None:
        code, diagnostics = _resolve_exit_code(
            True, "body\n\nACCEPTED\nnote: x", _NEXUS_MARKERS
        )
        assert code == ExitStatus.SUCCESS
        assert "marker=ACCEPTED" in diagnostics

    def test_bare_marker_rejected_usage_error(self) -> None:
        code, diagnostics = _resolve_exit_code(
            False, "REJECTED:missing_args", _NEXUS_MARKERS
        )
        assert code == ExitStatus.USAGE_ERROR
        assert "marker=REJECTED:missing_args" in diagnostics

    def test_prefix_match_rejected(self) -> None:
        assert _matches_expected("REJECTED:FILE_NOT_FOUND", ["REJECTED:"]) is True
        assert (
            _extract_status_marker("REJECTED:FILE_NOT_FOUND", ["REJECTED:"])
            == "REJECTED:FILE_NOT_FOUND"
        )

    def test_missing_marker_observe_vs_enforce(self) -> None:
        msg = "processed"
        expected = ["DONE"]
        code_obs, diag_obs = _resolve_exit_code(True, msg, expected, enforce=False)
        assert code_obs == ExitStatus.SUCCESS
        assert "contract_violation: PIPELINE_STATUS marker missing" in diag_obs
        assert f"expected_markers={expected}" in diag_obs

        code_enf, diag_enf = _resolve_exit_code(True, msg, expected, enforce=True)
        assert code_enf == ExitStatus.CONTRACT_VIOLATION
        assert "contract_violation: PIPELINE_STATUS marker missing" in diag_enf

    def test_undeclared_observe_vs_enforce(self) -> None:
        msg = "PIPELINE_STATUS: UNKNOWN_MARKER"
        expected = ["DONE"]
        code_obs, diag_obs = _resolve_exit_code(True, msg, expected, enforce=False)
        assert code_obs == ExitStatus.SUCCESS
        assert "violated_marker=UNKNOWN_MARKER" in diag_obs

        code_enf, diag_enf = _resolve_exit_code(True, msg, expected, enforce=True)
        assert code_enf == ExitStatus.CONTRACT_VIOLATION
        assert "violated_marker=UNKNOWN_MARKER" in diag_enf

    def test_empty_expected_ignores_enforce_and_bare(self) -> None:
        code, diagnostics = _resolve_exit_code(
            True, "ACCEPTED\n", [], enforce=True
        )
        assert code == ExitStatus.SUCCESS
        assert all(not d.startswith("marker=") for d in diagnostics)

    def test_fixture_bare_markers(self) -> None:
        for name, expected_marker in (
            ("claude_bare_marker.txt", "ACCEPTED"),
            ("cursor_bare_marker.txt", "ACCEPTED"),
            ("codex_bare_marker.txt", "ACCEPTED:NO_ACTION"),
        ):
            msg = (_FIXTURE_DIR / name).read_text(encoding="utf-8")
            code, diagnostics = _resolve_exit_code(True, msg, _NEXUS_MARKERS)
            assert code == ExitStatus.SUCCESS, name
            assert f"marker={expected_marker}" in diagnostics, name

    def test_fixture_no_markers_observe(self) -> None:
        for name in (
            "claude_no_marker.txt",
            "cursor_no_marker.txt",
            "codex_no_marker.txt",
        ):
            msg = (_FIXTURE_DIR / name).read_text(encoding="utf-8")
            code, diagnostics = _resolve_exit_code(True, msg, _NEXUS_MARKERS)
            assert code == ExitStatus.SUCCESS, name
            assert "marker=<absent>" in diagnostics, name
            assert (
                "contract_violation: PIPELINE_STATUS marker missing" in diagnostics
            ), name


class TestSkillResultDiagnosticsAndAudit:
    """Issue #3040: run_output.diagnostics and fanout skill_result audit."""

    def test_diagnostics_populated_in_run_output(self, tmp_path: Path) -> None:
        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path, status_markers=["DONE"])
        job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a"})
        (tmp_path / "jobs").mkdir(exist_ok=True)
        captured: dict = {}

        real_run = __import__(
            "mltgnt.skill.runner", fromlist=["run"]
        ).run

        def wrap_run(*args, **kwargs):
            result = real_run(*args, **kwargs)
            captured["run_output"] = result
            return result

        with (
            patch(_ENQUEUE, return_value=(True, "PIPELINE_STATUS: DONE")),
            patch("mltgnt.skill.runner.run", side_effect=wrap_run),
        ):
            ok, _ = run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert ok is True
        assert "run_output" in captured
        assert captured["run_output"].diagnostics
        assert captured["run_output"].exit_code == ExitStatus.SUCCESS

        audit_path = tmp_path / "jobs" / "audit.jsonl"
        records = [
            json.loads(line)
            for line in audit_path.read_text().splitlines()
            if line.strip()
            and json.loads(line).get("event_type") == "skill_result"
        ]
        assert len(records) == 1
        assert records[0]["exit_code"] == ExitStatus.SUCCESS
        assert records[0]["diagnostics"]

    def test_fanout_steps_audit_written(self, tmp_path: Path) -> None:
        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path)
        job = _skill_job(
            action_args={
                "skill": "test-skill",
                "persona": "persona-a",
                "enable_fanout": True,
            }
        )
        (tmp_path / "jobs").mkdir(exist_ok=True)

        with (
            patch(_ENQUEUE, return_value=(True, _FANOUT_RESPONSE)),
            patch(
                _ENQUEUE_DAG,
                return_value=[(True, ""), (False, "failed")],
            ),
        ):
            ok, msg = run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert ok is False
        assert "failed" in msg
        audit_path = tmp_path / "jobs" / "audit.jsonl"
        records = [
            json.loads(line)
            for line in audit_path.read_text().splitlines()
            if line.strip()
            and json.loads(line).get("event_type") == "skill_result"
        ]
        assert len(records) == 2
        assert "exit_code" in records[0]
        assert "diagnostics" in records[0]


class TestRunSkillActionPermissionPassthrough:
    def test_permission_none_when_not_in_action_args(self, tmp_path: Path) -> None:
        """When action_args has no permission key, enqueue_and_wait gets permission=None."""
        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path)
        job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a"})
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
        """action_args.permission='dangerous_full_access' is passed to enqueue_and_wait."""
        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path)
        job = _skill_job(action_args={
            "skill": "test-skill",
            "persona": "persona-a",
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


class TestResultFrontmatterOptIn:
    """Issue #3179 AC-4: result_frontmatter is opt-in via action_args."""

    def test_default_passes_run_result_none(self, tmp_path: Path) -> None:
        """No result_frontmatter key → enqueue_and_wait gets run_result=None"""
        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path, status_markers=["DONE"])
        job = _skill_job(action_args={"skill": "test-skill", "persona": "persona-a"})
        (tmp_path / "jobs").mkdir(exist_ok=True)
        captured_kwargs: dict = {}

        def capture_enqueue(**kwargs):
            captured_kwargs.update(kwargs)
            return True, "PIPELINE_STATUS: DONE"

        with patch(_ENQUEUE, side_effect=capture_enqueue):
            run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert "run_result" in captured_kwargs
        assert captured_kwargs["run_result"] is None

    def test_true_passes_run_output(self, tmp_path: Path) -> None:
        """result_frontmatter: true → enqueue_and_wait receives run_output"""
        persona_dir = _make_persona(tmp_path)
        meta = _make_skill_meta("test-skill", tmp_path, status_markers=["DONE"])
        job = _skill_job(
            action_args={
                "skill": "test-skill",
                "persona": "persona-a",
                "result_frontmatter": True,
            }
        )
        (tmp_path / "jobs").mkdir(exist_ok=True)
        captured_kwargs: dict = {}

        def capture_enqueue(**kwargs):
            captured_kwargs.update(kwargs)
            return True, "PIPELINE_STATUS: DONE"

        with patch(_ENQUEUE, side_effect=capture_enqueue):
            run_skill_action(
                job,
                persona_dir=persona_dir,
                skill_registry={"test-skill": meta},
                default_tz="Asia/Tokyo",
                repo_root=tmp_path,
            )

        assert captured_kwargs.get("run_result") is not None
        assert getattr(captured_kwargs["run_result"], "chat_input", None) is not None


class TestRunSkillActionExitCodeBranch:
    def test_success_returns_original_msg(self, tmp_path: Path) -> None:
        ok, msg = _run_skill(tmp_path, enqueue_return=(True, "response text"))
        assert ok is True
        assert msg == "response text"

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
            "persona": "persona-a",
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
