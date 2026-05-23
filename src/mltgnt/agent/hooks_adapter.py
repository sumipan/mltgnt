"""mltgnt.agent.hooks_adapter — AgentRunner.audit_writer 用コールバックファクトリ。"""
from __future__ import annotations

import uuid as _uuid_module
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ghdag.pipeline.audit import write_task_exit_audit


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
