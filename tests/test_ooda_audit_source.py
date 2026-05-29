"""AuditJsonlSource のユニットテスト。"""
from __future__ import annotations

import json

from mltgnt.ooda.audit_source import AuditJsonlSource


def _write_audit(path, records):
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_observe_returns_empty_when_file_missing(tmp_path):
    source = AuditJsonlSource(tmp_path / "missing.jsonl")
    assert source.observe() == []


def test_observe_filters_failed_status_and_prefix(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    _write_audit(
        audit_path,
        [
            {
                "uuid": "ok-1",
                "event_type": "task_complete",
                "status": "success",
                "timestamp": "2026-05-29T10:00:00+09:00",
                "idempotency_key": "failure-analysis:100",
            },
            {
                "uuid": "fail-1",
                "event_type": "task_failed",
                "status": "failure",
                "timestamp": "2026-05-29T10:01:00+09:00",
                "idempotency_key": "failure-analysis:100",
            },
            {
                "uuid": "fail-2",
                "event_type": "task_failed",
                "status": "failed",
                "timestamp": "2026-05-29T10:02:00+09:00",
                "idempotency_key": "other:1",
            },
        ],
    )
    source = AuditJsonlSource(audit_path, observe_filter="failure-analysis")
    events = source.observe()
    assert len(events) == 1
    assert events[0].event_id == "fail-1"
    assert events[0].status == "failure"


def test_observe_since_filters_by_timestamp(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    _write_audit(
        audit_path,
        [
            {
                "uuid": "old",
                "event_type": "task_failed",
                "status": "failed",
                "timestamp": "2026-05-29T09:00:00+09:00",
            },
            {
                "uuid": "new",
                "event_type": "task_failed",
                "status": "failed",
                "timestamp": "2026-05-29T11:00:00+09:00",
            },
        ],
    )
    source = AuditJsonlSource(audit_path)
    events = source.observe(since="2026-05-29T10:00:00+09:00")
    assert [event.event_id for event in events] == ["new"]
