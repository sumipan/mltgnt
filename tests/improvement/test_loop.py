from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest
from freezegun import freeze_time

from mltgnt.improvement import CycleResult, run_improvement_cycle


def _write_audit(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _touch(path: Path, text: str = "stub") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _four_category_audit_records() -> list[dict]:
    """4カテゴリ各 count>=3 相当の task_failed を含む audit レコード。"""
    records: list[dict] = []
    for _ in range(3):
        records.append(
            {
                "event_type": "task_failed",
                "correlation_id": "slack:triage-1",
                "persona": "タチコマ",
                "timestamp": "2026-05-28T10:00:00+09:00",
            }
        )
    records.extend(
        [
            {"event_type": "task_exit", "correlation_id": "slack:triage-1", "timestamp": "2026-05-28T10:01:00+09:00"},
            {"event_type": "task_exit", "correlation_id": "slack:triage-1", "timestamp": "2026-05-28T10:02:00+09:00"},
        ]
    )
    for i in range(3):
        records.append(
            {
                "event_type": "task_failed",
                "correlation_id": f"sched:timeout-{i}",
                "error": "upstream timeout reached",
                "persona": "タチコマ",
                "timestamp": "2026-05-28T11:00:00+09:00",
            }
        )
    for i in range(3):
        records.append(
            {
                "event_type": "task_failed",
                "correlation_id": f"agent:skill-{i}",
                "skill": "system-improve-agents",
                "persona": "タチコマ",
                "timestamp": "2026-05-28T12:00:00+09:00",
            }
        )
    for i in range(3):
        records.append(
            {
                "event_type": "task_failed",
                "correlation_id": f"agent:quality-{i}",
                "persona": "タチコマ",
                "timestamp": "2026-05-28T13:00:00+09:00",
            }
        )
    return records


@freeze_time("2026-05-29")
def test_run_improvement_cycle_returns_patterns_and_proposals(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    persona_dir = tmp_path / "personas"
    skills_dir = tmp_path / "skills"
    _touch(persona_dir / "タチコマ.md")
    _touch(skills_dir / "system-improve-agents" / "SKILL.md")
    _write_audit(audit_path, _four_category_audit_records())

    result = run_improvement_cycle(audit_path, persona_dir, skills_dir)

    assert isinstance(result, CycleResult)
    assert result.period_start == date(2026, 5, 22)
    assert result.period_end == date(2026, 5, 29)
    assert len(result.patterns) == 4
    assert len(result.proposals) >= 1


@freeze_time("2026-05-29")
def test_run_improvement_cycle_empty_audit(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    persona_dir = tmp_path / "personas"
    skills_dir = tmp_path / "skills"
    audit_path.write_text("", encoding="utf-8")

    result = run_improvement_cycle(audit_path, persona_dir, skills_dir)

    assert result.patterns == []
    assert result.proposals == []
    assert result.period_start == date(2026, 5, 22)
    assert result.period_end == date(2026, 5, 29)


@freeze_time("2026-05-29")
def test_run_improvement_cycle_since_days_zero(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text("", encoding="utf-8")

    result = run_improvement_cycle(
        audit_path, tmp_path / "p", tmp_path / "s", since_days=0
    )

    assert result.period_start == date(2026, 5, 29)
    assert result.period_end == date(2026, 5, 29)


@freeze_time("2026-05-29")
def test_run_improvement_cycle_default_since_days_seven(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text("", encoding="utf-8")

    result = run_improvement_cycle(audit_path, tmp_path / "p", tmp_path / "s")

    assert result.period_start == date.today() - timedelta(days=7)
    assert result.period_end == date.today()


def test_run_improvement_cycle_missing_audit_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_improvement_cycle(
            tmp_path / "missing.jsonl", tmp_path / "p", tmp_path / "s"
        )


def test_import_cycle_result_and_run_improvement_cycle() -> None:
    from mltgnt.improvement import CycleResult as CR, run_improvement_cycle as ric

    assert CR is CycleResult
    assert ric is run_improvement_cycle


def _run_improvement_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "mltgnt.improvement", *args]
    worktree = Path(__file__).resolve().parents[2]
    return subprocess.run(
        cmd,
        cwd=cwd or worktree,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )


@freeze_time("2026-05-29")
def test_cli_prints_markdown_report(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    persona_dir = tmp_path / "personas"
    skills_dir = tmp_path / "skills"
    _touch(persona_dir / "タチコマ.md")
    _touch(skills_dir / "system-improve-agents" / "SKILL.md")
    _write_audit(audit_path, _four_category_audit_records())

    worktree = Path(__file__).resolve().parents[2]
    result = _run_improvement_cli(
        "--audit",
        str(audit_path),
        "--persona-dir",
        str(persona_dir),
        "--skills-dir",
        str(skills_dir),
        cwd=worktree,
    )
    assert result.returncode == 0
    assert "# サマリ" in result.stdout
    assert "# 失敗パターン一覧" in result.stdout


def test_cli_missing_audit_exits_one(tmp_path: Path) -> None:
    worktree = Path(__file__).resolve().parents[2]
    result = _run_improvement_cli(
        "--audit",
        str(tmp_path / "missing.jsonl"),
        "--persona-dir",
        str(tmp_path),
        "--skills-dir",
        str(tmp_path),
        cwd=worktree,
    )
    assert result.returncode == 1
    assert result.stderr.strip()


def test_cli_missing_required_args_exits_two() -> None:
    worktree = Path(__file__).resolve().parents[2]
    result = _run_improvement_cli(cwd=worktree)
    assert result.returncode == 2
