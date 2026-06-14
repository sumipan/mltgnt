from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from mltgnt.improvement.patch import PatchResult
from mltgnt.kpi import KPIReport, compute_kpis

if TYPE_CHECKING:
    from mltgnt.improvement.loop import CycleResult

_METRICS = (
    ("response_failure_rate", "response_failure_rate"),
    ("re_question_rate", "re_question_rate"),
)


@dataclass
class RollbackDecision:
    should_rollback: bool
    reason: str
    degraded_metrics: list[str]


def evaluate_rollback(
    before: KPIReport,
    after: KPIReport,
    threshold: float = 0.05,
) -> RollbackDecision:
    degraded_metrics: list[str] = []
    reason_parts: list[str] = []

    for metric_name, attr in _METRICS:
        delta = getattr(after, attr) - getattr(before, attr)
        if delta > threshold:
            degraded_metrics.append(metric_name)
            reason_parts.append(
                f"{metric_name} degraded by {delta:.2f} (threshold: {threshold:.2f})"
            )

    if degraded_metrics:
        return RollbackDecision(
            should_rollback=True,
            reason="; ".join(reason_parts),
            degraded_metrics=degraded_metrics,
        )

    return RollbackDecision(
        should_rollback=False,
        reason=f"no metrics exceeded threshold (threshold: {threshold:.2f})",
        degraded_metrics=[],
    )


def _parse_pr_ref(pr_url: str) -> tuple[str, str, int]:
    marker = "/pull/"
    if marker not in pr_url:
        raise ValueError(f"invalid pr url: {pr_url}")
    repo_part, number_part = pr_url.rsplit(marker, 1)
    number = int(number_part.rstrip("/"))
    owner, repo = repo_part.rstrip("/").split("/")[-2:]
    return owner, repo, number


def execute_rollback(
    patch_results: list[PatchResult],
    repo_root: Path,
) -> list[str]:
    messages: list[str] = []

    for result in patch_results:
        if not result.applied or result.pr_url is None:
            continue

        pr_url = result.pr_url
        view = subprocess.run(
            ["gh", "pr", "view", pr_url, "--json", "state,number"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if view.returncode != 0:
            continue

        data = json.loads(view.stdout)
        state = data["state"]
        number = data["number"]

        if state == "OPEN":
            close = subprocess.run(
                ["gh", "pr", "close", pr_url],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
            if close.returncode == 0:
                messages.append(f"Closed PR #{number}")
        elif state == "MERGED":
            owner, repo, _ = _parse_pr_ref(pr_url)
            revert = subprocess.run(
                [
                    "gh",
                    "api",
                    "-X",
                    "POST",
                    f"repos/{owner}/{repo}/pulls/{number}/revert",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
            if revert.returncode == 0:
                messages.append(f"Created revert PR for #{number}")

    return messages


def evaluate_cycle_outcome(
    previous: CycleResult,
    audit_path: Path,
    repo_root: Path,
    *,
    today: date | None = None,
    window_days: int = 7,
    threshold: float = 0.05,
) -> RollbackDecision:
    """パッチ適用後の KPI 変化を評価してロールバック判定を返す。

    run_improvement_cycle(eval_rollback=True) の 2 ステップ目として呼び出す。
    パッチ適用から window_days 経過後に呼ぶことで意味ある判定ができる。

    previous.baseline_kpis が None の場合は ValueError を送出する。
    判定結果が should_rollback=True かつ previous.patch_results がある場合は
    execute_rollback() を自動実行する。
    """
    if previous.baseline_kpis is None:
        raise ValueError("previous cycle did not capture baseline KPIs")

    today = today or date.today()
    after = compute_kpis(
        audit_path,
        since=today - timedelta(days=window_days),
        until=today,
    )
    decision = evaluate_rollback(previous.baseline_kpis, after, threshold)

    if decision.should_rollback and previous.patch_results:
        execute_rollback(previous.patch_results, repo_root)

    return decision
