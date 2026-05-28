"""KPI 計算のユニットテスト。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from mltgnt.kpi import compute_kpis
from mltgnt.kpi._metrics import re_question_rate, response_failure_rate


def _write_audit(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def test_response_failure_rate_basic() -> None:
    records = [
        {"event_type": "task_complete", "correlation_id": "x"},
        {"event_type": "task_failed", "correlation_id": "x"},
        {"event_type": "task_complete", "correlation_id": "y"},
    ]
    rate, (failed, total) = response_failure_rate(records)
    assert rate == pytest.approx(1 / 3)
    assert failed == 1
    assert total == 3


def test_response_failure_rate_zero_failed() -> None:
    records = [{"event_type": "task_complete"}] * 5
    rate, (failed, total) = response_failure_rate(records)
    assert rate == 0.0
    assert failed == 0
    assert total == 5


def test_response_failure_excludes_dep_failed_and_rejected() -> None:
    records = [
        {"event_type": "task_complete"},
        {"event_type": "task_dep_failed"},
        {"event_type": "task_rejected"},
        {"event_type": "task_empty_result"},
    ]
    rate, (_, total) = response_failure_rate(records)
    assert rate == 0.0
    assert total == 1


def test_re_question_rate_slack_only() -> None:
    records = [
        {"event_type": "task_complete", "correlation_id": "slack:t1"},
        {"event_type": "task_failed", "correlation_id": "slack:t1"},
        {"event_type": "task_complete", "correlation_id": "slack:t2"},
        {"event_type": "task_complete", "correlation_id": "issuesmith:pipe"},
        {"event_type": "task_failed", "correlation_id": "issuesmith:pipe"},
        {"event_type": "task_complete", "correlation_id": "other"},
    ]
    rate, (retried, threads) = re_question_rate(records)
    assert threads == 2
    assert retried == 1
    assert rate == pytest.approx(0.5)


def test_compute_kpis_empty_file(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    audit.write_text("", encoding="utf-8")
    report = compute_kpis(audit)
    assert report.response_failure_rate == 0.0
    assert report.re_question_rate == 0.0
    assert report.response_failure_detail == (0, 0)
    assert report.skill_resolution_rate is None


def test_compute_kpis_since_filter(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    _write_audit(
        audit,
        [
            {
                "event_type": "task_complete",
                "timestamp": "2026-05-24T10:00:00+09:00",
                "correlation_id": "slack:a",
            },
            {
                "event_type": "task_failed",
                "timestamp": "2026-05-25T10:00:00+09:00",
                "correlation_id": "slack:b",
            },
        ],
    )
    report = compute_kpis(audit, since=date(2026, 5, 25))
    assert report.response_failure_detail == (1, 1)
    assert report.period_start == "2026-05-25"


def test_iter_audit_skips_invalid_lines(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    audit.write_text(
        '{"event_type": "task_complete", "timestamp": "2026-05-28T00:00:00+09:00"}\n'
        "not json\n"
        '{"event_type": "task_failed", "timestamp": "2026-05-28T01:00:00+09:00"}\n',
        encoding="utf-8",
    )
    report = compute_kpis(audit)
    assert report.response_failure_detail == (1, 2)
