from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from mltgnt.improvement.proposal import ImprovementProposal


@dataclass
class PatchResult:
    proposal_id: str
    applied: bool
    pr_url: str | None
    requires_human_review: bool
    reason: str


def _parse_diff_paths(diff_content: str) -> list[str]:
    paths: list[str] = []
    for line in diff_content.splitlines():
        if line.startswith("+++ b/"):
            paths.append(line[len("+++ b/") :])
    return paths


def _path_is_sensitive(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return (
        normalized.startswith("routing/")
        or normalized.startswith("daemon/")
        or "/routing/" in normalized
        or "/daemon/" in normalized
    )


def _is_low_risk(target_type: str, paths: list[str]) -> bool:
    if target_type not in ("persona", "trigger"):
        return False
    return not any(_path_is_sensitive(path) for path in paths)


def apply_proposal(
    proposal: ImprovementProposal,
    repo_root: Path,
) -> PatchResult:
    if proposal.diff_content is None:
        return PatchResult(
            proposal_id=proposal.proposal_id,
            applied=False,
            pr_url=None,
            requires_human_review=False,
            reason="no diff_content",
        )

    diff_content = proposal.diff_content

    dry_run = subprocess.run(
        ["patch", "-p1", "--dry-run"],
        input=diff_content,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if dry_run.returncode != 0:
        reason = (dry_run.stderr or dry_run.stdout or "patch dry-run failed").strip()
        return PatchResult(
            proposal_id=proposal.proposal_id,
            applied=False,
            pr_url=None,
            requires_human_review=False,
            reason=reason,
        )

    apply_result = subprocess.run(
        ["patch", "-p1"],
        input=diff_content,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if apply_result.returncode != 0:
        reason = (apply_result.stderr or apply_result.stdout or "patch apply failed").strip()
        return PatchResult(
            proposal_id=proposal.proposal_id,
            applied=False,
            pr_url=None,
            requires_human_review=False,
            reason=reason,
        )

    paths = _parse_diff_paths(diff_content)
    requires_human_review = not _is_low_risk(proposal.target_type, paths)

    pr_body = (
        f"Automated RSI proposal: {proposal.proposal_id}\n\n"
        f"{proposal.description}\n\n"
        f"Target: {proposal.target_type}/{proposal.target_name}"
    )
    pr_create = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--title",
            f"RSI: {proposal.description}",
            "--body",
            pr_body,
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if pr_create.returncode != 0:
        reason = (pr_create.stderr or pr_create.stdout or "gh pr create failed").strip()
        return PatchResult(
            proposal_id=proposal.proposal_id,
            applied=False,
            pr_url=None,
            requires_human_review=requires_human_review,
            reason=reason,
        )

    pr_url = pr_create.stdout.strip()
    return PatchResult(
        proposal_id=proposal.proposal_id,
        applied=True,
        pr_url=pr_url or None,
        requires_human_review=requires_human_review,
        reason="",
    )
