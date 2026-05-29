from __future__ import annotations

import json
from datetime import date

import pytest

from mltgnt.improvement import analyze_failures


def _write_audit(path, records):
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def test_analyze_failures_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_failures(tmp_path / "missing.jsonl")


def test_analyze_failures_empty_file_returns_empty(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text("", encoding="utf-8")
    assert analyze_failures(audit_path) == []


def test_analyze_failures_classifies_and_aggregates_patterns(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    records = [
        {
            "event_type": "task_failed",
            "correlation_id": "slack:triage-1",
            "timestamp": "2026-05-22T10:00:00+09:00",
        },
        {
            "event_type": "task_exit",
            "correlation_id": "slack:triage-1",
            "timestamp": "2026-05-22T10:01:00+09:00",
        },
        {
            "event_type": "task_exit",
            "correlation_id": "slack:triage-1",
            "timestamp": "2026-05-22T10:02:00+09:00",
        },
        {
            "event_type": "task_failed",
            "correlation_id": "sched:timeout-1",
            "error": "upstream timeout reached",
            "timestamp": "2026-05-23T12:00:00+09:00",
        },
        {
            "event_type": "task_failed",
            "correlation_id": "agent:skill-1",
            "skill": "system-improve-agents",
            "persona": "タチコマ",
            "timestamp": "2026-05-24T09:00:00+09:00",
        },
        {
            "event_type": "task_failed",
            "correlation_id": "agent:quality-1",
            "persona": "タチコマ",
            "timestamp": "2026-05-25T09:00:00+09:00",
        },
        {
            "event_type": "task_failed",
            "correlation_id": "agent:quality-1",
            "persona": "タチコマ",
            "timestamp": "2026-05-25T09:10:00+09:00",
        },
    ]
    _write_audit(audit_path, records)

    patterns = analyze_failures(audit_path, since=date(2026, 5, 21), until=date(2026, 5, 28))

    by_category = {pattern.category: pattern for pattern in patterns}
    assert set(by_category) == {"triage_error", "timeout", "skill_mismatch", "persona_quality"}
    assert by_category["triage_error"].count == 1
    assert by_category["timeout"].count == 1
    assert by_category["skill_mismatch"].count == 1
    assert by_category["persona_quality"].count == 2
    assert by_category["triage_error"].example_correlation_ids == ["slack:triage-1"]


def test_analyze_failures_applies_date_window(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    records = [
        {
            "event_type": "task_failed",
            "correlation_id": "agent:old",
            "timestamp": "2026-05-20T09:00:00+09:00",
        },
        {
            "event_type": "task_failed",
            "correlation_id": "agent:new",
            "timestamp": "2026-05-22T09:00:00+09:00",
        },
    ]
    _write_audit(audit_path, records)

    patterns = analyze_failures(audit_path, since=date(2026, 5, 21), until=date(2026, 5, 28))
    assert len(patterns) == 1
    assert patterns[0].example_correlation_ids == ["agent:new"]
