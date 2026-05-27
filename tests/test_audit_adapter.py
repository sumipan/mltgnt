from __future__ import annotations

import json
from pathlib import Path

from mltgnt.bridges.audit_adapter import (
    OrchestrationContext,
    end_orchestration,
    record_event,
    start_orchestration,
)


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_record_event_writes_jsonl(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    ctx = OrchestrationContext(orchestration_id="orch-test", source="test-source")

    record_event(audit_path, ctx, event_type="test_event", foo="bar")

    records = _read_records(audit_path)
    assert len(records) == 1
    record = records[0]
    assert record["event_type"] == "test_event"
    assert record["orchestration_id"] == "orch-test"
    assert record["source"] == "test-source"
    assert record["foo"] == "bar"
    assert record["schema_version"] == 2
    assert "timestamp" in record
    assert "uuid" in record


def test_record_persona_call(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    ctx = OrchestrationContext(orchestration_id="orch-call", source="slack_thread")

    ctx.record_persona_call(audit_path, engine="cursor", model="gemini-3-flash", ok=True)

    record = _read_records(audit_path)[0]
    assert record["event_type"] == "persona_call"
    assert record["engine"] == "cursor"
    assert record["model"] == "gemini-3-flash"
    assert record["ok"] is True


def test_audit_adapter_has_no_ghdag_import() -> None:
    import inspect

    import mltgnt.bridges.audit_adapter as mod

    source = inspect.getsource(mod)
    assert "from ghdag" not in source
    assert "import ghdag" not in source


def test_start_end_orchestration_events(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    ctx = OrchestrationContext(orchestration_id="orch-flow", source="test")

    start_orchestration(audit_path, ctx)
    end_orchestration(audit_path, ctx, status="success", elapsed_sec=1.23)

    records = _read_records(audit_path)
    assert records[0]["event_type"] == "orchestration_start"
    assert records[1]["event_type"] == "orchestration_end"
    assert records[1]["status"] == "success"
    assert records[1]["elapsed_sec"] == 1.23
