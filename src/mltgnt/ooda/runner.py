"""mltgnt.ooda.runner — OODA サイクル実行。"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from mltgnt.agent import AgentRunner
from mltgnt.interfaces.ooda import (
    ActDispatcher,
    ActResult,
    OODAConfig,
    ObserveSource,
    ObservationEvent,
    OODATickResult,
)

_JST = ZoneInfo("Asia/Tokyo")
_OODA_DEDUPE_RE = re.compile(r"ooda:([^:]+):(\d+)")


class OODARunner:
    """Observe → Orient → Decide → Act → Feedback の 1 tick を実行する。"""

    def __init__(
        self,
        *,
        observe_source: ObserveSource,
        agent_runner: AgentRunner,
        act_dispatcher: ActDispatcher,
        memory_append: Callable[..., None],
        memory_read: Callable[..., str],
        config: OODAConfig | None = None,
        audit_writer: Callable[[str, dict, str], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._observe_source = observe_source
        self._agent_runner = agent_runner
        self._act_dispatcher = act_dispatcher
        self._memory_append = memory_append
        self._memory_read = memory_read
        self._config = config if config is not None else OODAConfig()
        self._audit_writer = audit_writer
        self._logger = logger or logging.getLogger(__name__)
        self._last_since: str | None = None

    def run_tick(self) -> OODATickResult:
        events = self._observe_source.observe(since=self._last_since)
        self._last_since = _max_timestamp(events) or self._last_since

        actionable = [
            event
            for event in events
            if self._count_attempts(event.event_id) < self._config.max_recovery_attempts
        ]

        if not actionable:
            return OODATickResult(
                observed_events=len(events),
                actions_taken=[],
                escalated=False,
            )

        memory_context = self._memory_read()
        actions_taken: list[ActResult] = []
        escalated = False

        for event in actionable:
            attempts = self._count_attempts(event.event_id)
            if attempts >= self._config.escalate_after:
                act_result = self._act_dispatcher.dispatch(
                    "escalate_to_slack",
                    self._escalation_args(event),
                )
                escalated = True
            else:
                prompt = self._build_orient_prompt([event], memory_context)
                decision = self._agent_runner.run(prompt)
                if decision is None:
                    self._logger.warning(
                        "OODARunner: AgentRunner returned None for event %s",
                        event.event_id,
                    )
                    continue
                try:
                    act_result = self._act_dispatcher.dispatch(decision.tool, decision.args)
                except Exception as exc:
                    self._logger.error(
                        "OODARunner: ActDispatcher raised for event %s: %s",
                        event.event_id,
                        exc,
                    )
                    act_result = ActResult(
                        action=decision.tool,
                        success=False,
                        detail=str(exc),
                    )

            attempt_number = attempts + 1
            self._record_feedback(event, act_result, attempt_number)
            actions_taken.append(act_result)
            memory_context = self._memory_read()

        return OODATickResult(
            observed_events=len(events),
            actions_taken=actions_taken,
            escalated=escalated,
        )

    def _build_orient_prompt(
        self,
        events: list[ObservationEvent],
        memory_context: str,
    ) -> str:
        lines = [
            "You are an OODA recovery agent. Choose a terminal tool:",
            "recover_task, escalate_to_slack, or skip.",
            "",
            "## Observed failure events",
        ]
        for event in events:
            lines.append(
                f"- event_id={event.event_id} type={event.event_type} status={event.status} "
                f"timestamp={event.timestamp}"
            )
            lines.append(f"  payload={json.dumps(event.payload, ensure_ascii=False)}")
        lines.append("")
        lines.append("## Prior recovery attempts (memory)")
        lines.append(memory_context or "(none)")
        lines.append("")
        lines.append('Respond with JSON: {"tool": "<name>", "args": {...}}')
        return "\n".join(lines)

    def _count_attempts(self, event_id: str) -> int:
        memory_context = self._memory_read()
        if not memory_context:
            return 0

        count = 0
        for match in _OODA_DEDUPE_RE.finditer(memory_context):
            if match.group(1) == event_id:
                count += 1

        marker = f'"event_id": "{event_id}"'
        if marker in memory_context:
            count = max(count, memory_context.count(marker))

        return count

    def _record_feedback(
        self,
        event: ObservationEvent,
        act_result: ActResult,
        attempt_number: int,
    ) -> None:
        timestamp = datetime.now(tz=_JST).isoformat()
        content = json.dumps(
            {
                "event_id": event.event_id,
                "attempt": attempt_number,
                "action": act_result.action,
                "success": act_result.success,
                "detail": act_result.detail,
            },
            ensure_ascii=False,
        )
        dedupe_key = f"ooda:{event.event_id}:{attempt_number}"
        self._memory_append(
            role="system",
            content=content,
            timestamp=timestamp,
            source_tag="ooda_feedback",
            layer="ooda",
            dedupe_key=dedupe_key,
        )
        if self._audit_writer is not None:
            try:
                self._audit_writer(
                    "ooda_act",
                    {
                        "event_id": event.event_id,
                        "action": act_result.action,
                        "success": act_result.success,
                        "detail": act_result.detail,
                        "attempt": attempt_number,
                    },
                    act_result.detail,
                )
            except Exception as exc:
                self._logger.warning("OODARunner: audit_writer raised: %s", exc)

    def _escalation_args(self, event: ObservationEvent) -> dict[str, Any]:
        return {
            "text": (
                f"OODA recovery escalated for event {event.event_id} "
                f"({event.event_type}, status={event.status})"
            ),
            "channel": event.payload.get("slack_channel", ""),
            "event_id": event.event_id,
        }


def _max_timestamp(events: list[ObservationEvent]) -> str | None:
    timestamps = [event.timestamp for event in events if event.timestamp]
    if not timestamps:
        return None
    return max(timestamps)
