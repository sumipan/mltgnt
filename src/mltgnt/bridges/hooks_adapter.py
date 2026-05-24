"""mltgnt.bridges.hooks_adapter — AgentRunner.audit_writer 用コールバックファクトリ + DagHooks adapter。"""
from __future__ import annotations

import uuid as _uuid_module
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ghdag.dag._util import check_pipeline_status, default_check_rejected
from ghdag.pipeline.audit import write_task_exit_audit

if TYPE_CHECKING:
    from ghdag.dag.hooks import Task, TaskMetrics


class MltgntHooks:
    """mltgnt 用 DagHooks 実装。DagHooks Protocol の全 10 メソッドを実装する。"""

    def __init__(self, audit_path: Path, *, source: str = "mltgnt-scheduler") -> None:
        self._audit_path = audit_path
        self._source = source

    def on_task_start(self, uuid: str, task: "Task") -> None:
        write_task_exit_audit(
            self._audit_path,
            event_type="task_started",
            uuid=uuid,
            status="running",
            engine=self._source,
            model=task.model,
        )

    def on_task_success(self, uuid: str, task: "Task", metrics: "TaskMetrics") -> None:
        write_task_exit_audit(
            self._audit_path,
            event_type="task_success",
            uuid=uuid,
            status="success",
            engine=metrics.engine or self._source,
            model=metrics.model,
            elapsed_sec=metrics.wall_time_sec,
            token_count=metrics.token_count,
            correlation_id=metrics.correlation_id,
        )

    def on_task_failure(
        self,
        uuid: str,
        task: "Task",
        returncode: int,
        stderr_text: str,
        metrics: "TaskMetrics",
    ) -> None:
        write_task_exit_audit(
            self._audit_path,
            event_type="task_failure",
            uuid=uuid,
            status="failure",
            engine=metrics.engine or self._source,
            model=metrics.model,
            elapsed_sec=metrics.wall_time_sec,
            token_count=metrics.token_count,
            correlation_id=metrics.correlation_id,
        )

    def on_task_rejected(
        self,
        uuid: str,
        task: "Task",
        retry_depth: int,
        is_final: bool,
        metrics: "TaskMetrics",
    ) -> None:
        write_task_exit_audit(
            self._audit_path,
            event_type="task_rejected",
            uuid=uuid,
            status="rejected",
            engine=metrics.engine or self._source,
            model=metrics.model,
            elapsed_sec=metrics.wall_time_sec,
            token_count=metrics.token_count,
            correlation_id=metrics.correlation_id,
        )

    def on_task_dep_failed(self, uuid: str, task: "Task", failed_dep: str) -> None:
        write_task_exit_audit(
            self._audit_path,
            event_type="task_dep_failed",
            uuid=uuid,
            status="dep_failed",
            engine=self._source,
            model=task.model,
            correlation_id=failed_dep,
        )

    def on_task_empty_result(
        self,
        uuid: str,
        task: "Task",
        stderr_text: str,
        metrics: "TaskMetrics",
    ) -> None:
        write_task_exit_audit(
            self._audit_path,
            event_type="task_empty_result",
            uuid=uuid,
            status="empty_result",
            engine=metrics.engine or self._source,
            model=metrics.model,
            elapsed_sec=metrics.wall_time_sec,
            token_count=metrics.token_count,
            correlation_id=metrics.correlation_id,
        )

    def on_shutdown(self, signum: int) -> None:
        write_task_exit_audit(
            self._audit_path,
            event_type="shutdown",
            uuid=str(_uuid_module.uuid4()),
            status="shutdown",
            engine=self._source,
            model=None,
            correlation_id=str(signum),
        )

    def check_rejected(self, result_path: str) -> bool:
        return default_check_rejected(result_path)

    def check_pipeline_status(self, result_path: str) -> "str | None":
        return check_pipeline_status(result_path)

    def check_promote_target(self, result_path: str) -> "str | None":
        return None


def create_audit_writer(
    audit_path: Path,
    *,
    source: str = "mltgnt-agent",
) -> Callable[[str, dict[str, Any], str], None]:
    """AgentRunner.audit_writer 用コールバックを返す。

    ツール実行ごとに write_task_exit_audit() を呼び出し、
    audit.jsonl に event_type="tool_exec" のレコードを追記する。
    tool_args の内容はレコードに含めない（機密データ混入防止）。
    """
    def _write(tool_name: str, tool_args: dict[str, Any], tool_result: str) -> None:
        write_task_exit_audit(
            audit_path,
            event_type="tool_exec",
            uuid=str(_uuid_module.uuid4()),
            status="success",
            engine=source,
            model=None,
            correlation_id=tool_name,
            elapsed_sec=None,
            token_count=None,
        )

    return _write
