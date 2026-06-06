from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from mltgnt.improvement import PatchResult
from mltgnt.improvement.patch import _is_low_risk, _parse_diff_paths, apply_proposal
from mltgnt.improvement.proposal import ImprovementProposal

_REAL_SUBPROCESS_RUN = subprocess.run


def _proposal(
    *,
    proposal_id: str = "proposal:test",
    target_type: str = "persona",
    target_name: str = "タチコマ",
    description: str = "テスト提案",
    diff_content: str | None = None,
) -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id=proposal_id,
        target_type=target_type,
        target_name=target_name,
        action="adjust_section",
        description=description,
        diff_preview="+ added",
        confidence=0.8,
        source_patterns=["test"],
        diff_content=diff_content,
    )


def _make_unified_diff(rel_path: str, original_line: str, added_line: str) -> str:
    return (
        f"--- a/{rel_path}\n"
        f"+++ b/{rel_path}\n"
        f"@@ -1 +1,2 @@\n"
        f" {original_line}\n"
        f"+{added_line}\n"
    )


def _touch(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_parse_diff_paths_extracts_b_paths():
    diff = _make_unified_diff("personas/foo.md", "# hello", "added")
    assert _parse_diff_paths(diff) == ["personas/foo.md"]


def test_is_low_risk_persona_outside_routing_daemon():
    assert _is_low_risk("persona", ["personas/foo.md", "skills/bar/SKILL.md"]) is True


def test_is_low_risk_trigger_outside_routing_daemon():
    assert _is_low_risk("trigger", ["skills/bar/SKILL.md"]) is True


def test_is_low_risk_false_for_routing_path():
    assert _is_low_risk("persona", ["routing/router.py"]) is False


def test_is_low_risk_false_for_daemon_path():
    assert _is_low_risk("persona", ["daemon/worker.py"]) is False


def test_is_low_risk_false_for_unknown_target_type():
    assert _is_low_risk("config", ["personas/foo.md"]) is False


def test_apply_proposal_returns_no_diff_content():
    result = apply_proposal(_proposal(diff_content=None), Path("/tmp"))
    assert result == PatchResult(
        proposal_id="proposal:test",
        applied=False,
        pr_url=None,
        requires_human_review=False,
        reason="no diff_content",
    )


def test_apply_proposal_dry_run_failure():
    repo_root = Path("/tmp/repo")
    invalid_diff = "not a valid unified diff"
    with patch("mltgnt.improvement.patch.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["patch"],
            returncode=1,
            stdout="",
            stderr="patch: **** malformed patch",
        )
        result = apply_proposal(_proposal(diff_content=invalid_diff), repo_root)

    assert result.applied is False
    assert result.pr_url is None
    assert result.requires_human_review is False
    assert "malformed patch" in result.reason
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == ["patch", "-p1", "--dry-run"]


def test_apply_proposal_apply_failure_after_dry_run_success(tmp_path: Path):
    rel_path = "personas/foo.md"
    _touch(tmp_path / rel_path, "# hello\n")
    diff = _make_unified_diff(rel_path, "# hello", "added")

    call_count = 0

    def fake_run(cmd, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="patch: **** failed",
        )

    with patch("mltgnt.improvement.patch.subprocess.run", side_effect=fake_run):
        result = apply_proposal(_proposal(diff_content=diff), tmp_path)

    assert result.applied is False
    assert "failed" in result.reason
    assert call_count == 2


def test_apply_proposal_success_applies_patch_and_creates_pr(tmp_path: Path):
    rel_path = "personas/foo.md"
    target_file = tmp_path / rel_path
    _touch(target_file, "# hello\n")
    diff = _make_unified_diff(rel_path, "# hello", "added line")

    pr_url = "https://github.com/sumipan/mltgnt/pull/999"
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[0] == "gh":
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=f"{pr_url}\n",
                stderr="",
            )
        return _REAL_SUBPROCESS_RUN(cmd, **kwargs)

    proposal = _proposal(
        target_type="persona",
        description="persona 改善",
        diff_content=diff,
    )
    with patch("mltgnt.improvement.patch.subprocess.run", side_effect=fake_run):
        result = apply_proposal(proposal, tmp_path)

    assert result.applied is True
    assert result.pr_url == pr_url
    assert result.requires_human_review is False
    assert result.reason == ""
    assert "# hello\nadded line\n" == target_file.read_text(encoding="utf-8")
    assert calls[-1][0:3] == ["gh", "pr", "create"]
    assert calls[-1][calls[-1].index("--title") + 1] == "RSI: persona 改善"


@pytest.mark.parametrize(
    ("target_type", "rel_path"),
    [
        ("persona", "personas/foo.md"),
        ("trigger", "skills/bar/SKILL.md"),
    ],
)
def test_apply_proposal_low_risk_target_types(tmp_path: Path, target_type: str, rel_path: str):
    _touch(tmp_path / rel_path, "# hello\n")
    diff = _make_unified_diff(rel_path, "# hello", "added")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "gh":
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="https://github.com/sumipan/mltgnt/pull/1\n",
                stderr="",
            )
        return _REAL_SUBPROCESS_RUN(cmd, **kwargs)

    with patch("mltgnt.improvement.patch.subprocess.run", side_effect=fake_run):
        result = apply_proposal(
            _proposal(target_type=target_type, diff_content=diff),
            tmp_path,
        )

    assert result.applied is True
    assert result.requires_human_review is False


@pytest.mark.parametrize(
    "rel_path",
    ["routing/router.py", "daemon/worker.py"],
)
def test_apply_proposal_persona_routing_or_daemon_requires_review(
    tmp_path: Path,
    rel_path: str,
):
    _touch(tmp_path / rel_path, "# hello\n")
    diff = _make_unified_diff(rel_path, "# hello", "added")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "gh":
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="https://github.com/sumipan/mltgnt/pull/2\n",
                stderr="",
            )
        return _REAL_SUBPROCESS_RUN(cmd, **kwargs)

    with patch("mltgnt.improvement.patch.subprocess.run", side_effect=fake_run):
        result = apply_proposal(
            _proposal(target_type="persona", diff_content=diff),
            tmp_path,
        )

    assert result.applied is True
    assert result.requires_human_review is True


def test_apply_proposal_unknown_target_type_requires_review(tmp_path: Path):
    rel_path = "config/settings.yaml"
    _touch(tmp_path / rel_path, "key: value\n")
    diff = _make_unified_diff(rel_path, "key: value", "key2: value2")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "gh":
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="https://github.com/sumipan/mltgnt/pull/3\n",
                stderr="",
            )
        return _REAL_SUBPROCESS_RUN(cmd, **kwargs)

    with patch("mltgnt.improvement.patch.subprocess.run", side_effect=fake_run):
        result = apply_proposal(
            _proposal(target_type="config", diff_content=diff),
            tmp_path,
        )

    assert result.applied is True
    assert result.requires_human_review is True


def test_patch_result_exported_from_improvement_package():
    from mltgnt.improvement import PatchResult as ExportedPatchResult

    assert ExportedPatchResult is PatchResult
