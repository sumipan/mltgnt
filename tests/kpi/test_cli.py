"""KPI CLI 出力テスト。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_kpi(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "mltgnt.kpi", *args]
    return subprocess.run(
        cmd,
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_text_output(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    audit.write_text(
        json.dumps(
            {
                "event_type": "task_complete",
                "timestamp": "2026-05-28T12:00:00+09:00",
                "correlation_id": "slack:thread-1",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    worktree = Path(__file__).resolve().parents[2]
    result = _run_kpi(str(audit), cwd=worktree)
    assert result.returncode == 0
    assert "=== mltgnt KPI Report ===" in result.stdout
    assert "Response Failure Rate:" in result.stdout
    assert "Skill Resolution Rate:  N/A" in result.stdout


def test_cli_json_output(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    audit.write_text(
        json.dumps(
            {
                "event_type": "task_failed",
                "timestamp": "2026-05-28T12:00:00+09:00",
                "correlation_id": "slack:t",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    worktree = Path(__file__).resolve().parents[2]
    result = _run_kpi(str(audit), "--format", "json", cwd=worktree)
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "response_failure_rate" in payload
    assert payload["response_failure_rate"] == 1.0


def test_cli_missing_file(tmp_path: Path) -> None:
    worktree = Path(__file__).resolve().parents[2]
    missing = tmp_path / "missing.jsonl"
    result = _run_kpi(str(missing), cwd=worktree)
    assert result.returncode == 1
    assert "audit file not found" in result.stderr
    assert "Traceback" not in result.stderr
