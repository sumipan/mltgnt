"""KPI R1 メトリクス（memory_recall_rate / task_completion_time_ms）のユニットテスト。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mltgnt.kpi import compute_kpis
from mltgnt.kpi._metrics import memory_recall_rate, task_completion_time_ms


def test_memory_recall_rate_hit_and_miss() -> None:
    records = [{"event_type": "memory_hit"}, {"event_type": "memory_miss"}]
    rate, (hit, total) = memory_recall_rate(records)
    assert rate == 0.5
    assert (hit, total) == (1, 2)


def test_memory_recall_rate_empty() -> None:
    rate, (hit, total) = memory_recall_rate([])
    assert rate == 0.0
    assert (hit, total) == (0, 0)


def test_memory_recall_rate_hit_only() -> None:
    records = [{"event_type": "memory_hit"}]
    rate, (hit, total) = memory_recall_rate(records)
    assert rate == 1.0
    assert (hit, total) == (1, 1)


def test_task_completion_time_ms_basic() -> None:
    records = [
        {"event_type": "task_complete", "elapsed_sec": 10.0},
        {"event_type": "task_complete", "elapsed_sec": 20.0},
    ]
    assert task_completion_time_ms(records) == 15000.0


def test_task_completion_time_ms_no_events() -> None:
    records = [{"event_type": "task_complete"}]
    assert task_completion_time_ms(records) is None


def test_task_completion_time_ms_skips_none_elapsed_sec() -> None:
    records = [
        {"event_type": "task_complete", "elapsed_sec": None},
        {"event_type": "task_complete", "elapsed_sec": 10.0},
    ]
    assert task_completion_time_ms(records) == 10000.0


def _write_audit(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def test_compute_kpis_memory_recall_rate(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    _write_audit(
        audit,
        [
            {"event_type": "memory_hit", "timestamp": "2026-05-28T12:00:00+09:00"},
            {"event_type": "memory_miss", "timestamp": "2026-05-28T12:01:00+09:00"},
        ],
    )
    report = compute_kpis(audit)
    assert report.memory_recall_rate == pytest.approx(0.5)


def test_compute_kpis_memory_recall_rate_no_data(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    _write_audit(audit, [{"event_type": "task_complete", "timestamp": "2026-05-28T12:00:00+09:00"}])
    report = compute_kpis(audit)
    assert report.memory_recall_rate is None


def test_compute_kpis_task_completion_time_ms(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    _write_audit(
        audit,
        [
            {
                "event_type": "task_complete",
                "elapsed_sec": 10.0,
                "timestamp": "2026-05-28T12:00:00+09:00",
            },
            {
                "event_type": "task_failed",
                "elapsed_sec": 20.0,
                "timestamp": "2026-05-28T12:01:00+09:00",
            },
        ],
    )
    report = compute_kpis(audit)
    assert report.task_completion_time_ms == 15000.0
