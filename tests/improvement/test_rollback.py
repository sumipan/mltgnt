from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from mltgnt.improvement import RollbackDecision
from mltgnt.improvement.patch import PatchResult
from mltgnt.improvement.rollback import evaluate_rollback, execute_rollback
from mltgnt.kpi import KPIReport


def _kpi(response_failure_rate: float, re_question_rate: float) -> KPIReport:
    return KPIReport(
        period_start="2026-05-01",
        period_end="2026-05-07",
        response_failure_rate=response_failure_rate,
        response_failure_detail=(0, 0),
        re_question_rate=re_question_rate,
        re_question_detail=(0, 0),
        skill_resolution_rate=None,
    )


def test_evaluate_rollback_response_failure_rate_degraded():
    before = _kpi(0.10, 0.05)
    after = _kpi(0.18, 0.05)

    decision = evaluate_rollback(before, after)

    assert decision.should_rollback is True
    assert decision.degraded_metrics == ["response_failure_rate"]
    assert "response_failure_rate degraded by 0.08" in decision.reason
    assert "(threshold: 0.05)" in decision.reason


def test_evaluate_rollback_re_question_rate_degraded():
    before = _kpi(0.10, 0.05)
    after = _kpi(0.10, 0.11)

    decision = evaluate_rollback(before, after)

    assert decision.should_rollback is True
    assert decision.degraded_metrics == ["re_question_rate"]
    assert "re_question_rate degraded by 0.06" in decision.reason


def test_evaluate_rollback_both_metrics_degraded():
    before = _kpi(0.10, 0.05)
    after = _kpi(0.20, 0.15)

    decision = evaluate_rollback(before, after)

    assert decision.should_rollback is True
    assert decision.degraded_metrics == ["response_failure_rate", "re_question_rate"]


def test_evaluate_rollback_no_degradation():
    before = _kpi(0.10, 0.05)
    after = _kpi(0.14, 0.09)

    decision = evaluate_rollback(before, after)

    assert decision.should_rollback is False
    assert decision.degraded_metrics == []
    assert "no metrics exceeded threshold" in decision.reason


def test_evaluate_rollback_at_threshold_boundary_no_rollback():
    before = _kpi(0.10, 0.05)
    after = _kpi(0.15, 0.10)

    decision = evaluate_rollback(before, after)

    assert decision.should_rollback is False
    assert decision.degraded_metrics == []


def test_execute_rollback_closes_open_pr(tmp_path: Path):
    pr_url = "https://github.com/sumipan/mltgnt/pull/42"
    patch_results = [
        PatchResult(
            proposal_id="proposal:1",
            applied=True,
            pr_url=pr_url,
            requires_human_review=False,
            reason="",
        )
    ]

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps({"state": "OPEN", "number": 42}),
                stderr="",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("mltgnt.improvement.rollback.subprocess.run", side_effect=fake_run):
        messages = execute_rollback(patch_results, tmp_path)

    assert messages == ["Closed PR #42"]
    assert ["gh", "pr", "close", pr_url] in calls


def test_execute_rollback_creates_revert_for_merged_pr(tmp_path: Path):
    pr_url = "https://github.com/sumipan/mltgnt/pull/41"
    patch_results = [
        PatchResult(
            proposal_id="proposal:2",
            applied=True,
            pr_url=pr_url,
            requires_human_review=False,
            reason="",
        )
    ]

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps({"state": "MERGED", "number": 41}),
                stderr="",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("mltgnt.improvement.rollback.subprocess.run", side_effect=fake_run):
        messages = execute_rollback(patch_results, tmp_path)

    assert messages == ["Created revert PR for #41"]
    assert [
        "gh",
        "api",
        "-X",
        "POST",
        "repos/sumipan/mltgnt/pulls/41/revert",
    ] in calls


def test_execute_rollback_skips_not_applied(tmp_path: Path):
    patch_results = [
        PatchResult(
            proposal_id="proposal:3",
            applied=False,
            pr_url="https://github.com/sumipan/mltgnt/pull/99",
            requires_human_review=False,
            reason="no diff_content",
        ),
        PatchResult(
            proposal_id="proposal:4",
            applied=True,
            pr_url=None,
            requires_human_review=False,
            reason="gh failed",
        ),
    ]

    with patch("mltgnt.improvement.rollback.subprocess.run") as mock_run:
        messages = execute_rollback(patch_results, tmp_path)

    assert messages == []
    mock_run.assert_not_called()


def test_rollback_decision_exported_from_improvement_package():
    from mltgnt.improvement import RollbackDecision as ExportedRollbackDecision

    assert ExportedRollbackDecision is RollbackDecision


@freeze_time("2026-05-29")
def test_run_improvement_cycle_eval_rollback_false_unchanged(tmp_path: Path) -> None:
    from mltgnt.improvement import run_improvement_cycle

    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text("", encoding="utf-8")

    result = run_improvement_cycle(
        audit_path,
        tmp_path / "personas",
        tmp_path / "skills",
    )

    assert result.patch_results is None
    assert result.rollback_decision is None


def test_run_improvement_cycle_eval_rollback_true_requires_repo_root(tmp_path: Path) -> None:
    from mltgnt.improvement import run_improvement_cycle

    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="repo_root is required when eval_rollback=True"):
        run_improvement_cycle(
            audit_path,
            tmp_path / "personas",
            tmp_path / "skills",
            eval_rollback=True,
        )
