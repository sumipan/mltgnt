"""mltgnt.ooda.audit_source — audit.jsonl から失敗イベントを観測する。"""
from __future__ import annotations

from pathlib import Path

from mltgnt.interfaces.ooda import ObservationEvent, ObserveSource
from mltgnt.kpi._parser import iter_audit_records

_FAILED_STATUSES = frozenset({"failed", "failure"})


def _record_to_event(record: dict) -> ObservationEvent | None:
    status = record.get("status")
    if not isinstance(status, str) or status not in _FAILED_STATUSES:
        return None

    event_id = record.get("uuid") or record.get("correlation_id")
    if not isinstance(event_id, str) or not event_id:
        return None

    event_type = record.get("event_type")
    timestamp = record.get("timestamp")
    return ObservationEvent(
        event_id=event_id,
        event_type=event_type if isinstance(event_type, str) else "unknown",
        status=status,
        timestamp=timestamp if isinstance(timestamp, str) else "",
        payload=dict(record),
    )


def _matches_since(timestamp: str, since: str | None) -> bool:
    if since is None:
        return True
    if not timestamp:
        return False
    return timestamp > since


def _matches_idempotency_filter(record: dict, prefix: str | None) -> bool:
    if prefix is None:
        return True
    key = record.get("idempotency_key")
    if isinstance(key, str) and key.startswith(prefix):
        return True
    payload_key = record.get("payload", {})
    if isinstance(payload_key, dict):
        nested = payload_key.get("idempotency_key")
        if isinstance(nested, str) and nested.startswith(prefix):
            return True
    return False


class AuditJsonlSource:
    """audit.jsonl を読み、失敗イベントを ObservationEvent として返す。"""

    def __init__(self, audit_path: Path, *, observe_filter: str | None = None) -> None:
        self._audit_path = audit_path
        self._observe_filter = observe_filter

    def observe(self, *, since: str | None = None) -> list[ObservationEvent]:
        if not self._audit_path.exists():
            return []

        events: list[ObservationEvent] = []
        for record in iter_audit_records(self._audit_path):
            if not _matches_idempotency_filter(record, self._observe_filter):
                continue
            event = _record_to_event(record)
            if event is None:
                continue
            if not _matches_since(event.timestamp, since):
                continue
            events.append(event)
        return events
