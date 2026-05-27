from __future__ import annotations

import json
import uuid as _uuid_module
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class OrchestrationContext:
    orchestration_id: str
    source: str
    parent_correlation_id: str | None = None

    def record_persona_call(
        self,
        audit_path: Path,
        *,
        engine: str | None = None,
        model: str | None = None,
        ok: bool | None = None,
    ) -> None:
        record_event(
            audit_path,
            self,
            event_type="persona_call",
            engine=engine,
            model=model,
            ok=ok,
        )


def record_event(
    audit_path: Path,
    ctx: OrchestrationContext,
    *,
    event_type: str,
    **fields: Any,
) -> None:
    payload = {
        "event_type": event_type,
        "orchestration_id": ctx.orchestration_id,
        "source": ctx.source,
        "parent_correlation_id": ctx.parent_correlation_id,
        "timestamp": datetime.now(tz=ZoneInfo("Asia/Tokyo")).isoformat(),
        "uuid": str(_uuid_module.uuid4()),
        "schema_version": 2,
    }
    payload.update(fields)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def start_orchestration(audit_path: Path, ctx: OrchestrationContext) -> None:
    record_event(audit_path, ctx, event_type="orchestration_start")


def end_orchestration(
    audit_path: Path,
    ctx: OrchestrationContext,
    *,
    status: str,
    elapsed_sec: float | None = None,
) -> None:
    record_event(
        audit_path,
        ctx,
        event_type="orchestration_end",
        status=status,
        elapsed_sec=elapsed_sec,
    )
