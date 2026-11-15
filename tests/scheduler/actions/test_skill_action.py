"""Unit tests for run_skill_action / _determine_exit_code (Issue #2076)."""
from __future__ import annotations

from pathlib import Path

from mltgnt.scheduler.actions.skill import (
    _determine_exit_code,
    _extract_status_marker,
    _matches_expected,
    _resolve_exit_code,
)
from mltgnt.skill.models import ExitStatus

from .skill_helpers import _run_skill

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
