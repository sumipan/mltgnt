"""OODARunner のユニットテスト（AC1〜AC7, AC10）。"""
from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from mltgnt.agent import AgentResult, AgentRunner
from mltgnt.interfaces.ooda import ActResult, OODAConfig, ObservationEvent
from mltgnt.ooda.audit_source import AuditJsonlSource
from mltgnt.ooda.exec_dispatcher import ExecAppenderDispatcher
from mltgnt.ooda.runner import OODARunner


class _MemoryStore:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def append(self, **kwargs: Any) -> None:
        self.entries.append(kwargs)

    def read_text(self) -> str:
        return "\n".join(
            json.dumps(entry, ensure_ascii=False) for entry in self.entries
        )


class _ListObserveSource:
    def __init__(self, events: list[ObservationEvent]) -> None:
        self._events = events

    def observe(self, *, since: str | None = None) -> list[ObservationEvent]:
        if since is None:
            return list(self._events)
        return [event for event in self._events if event.timestamp > since]


class _RecordingDispatcher:
    def __init__(self, results: dict[str, ActResult] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._results = results or {}

    def dispatch(self, action: str, args: dict[str, Any]) -> ActResult:
        self.calls.append((action, args))
        if action in self._results:
            return self._results[action]
        return ActResult(action=action, success=True, detail="ok")


def _failure_event(
    event_id: str = "evt-1",
    *,
    timestamp: str = "2026-05-29T10:00:00+09:00",
) -> ObservationEvent:
    return ObservationEvent(
        event_id=event_id,
        event_type="task_failed",
        status="failed",
        timestamp=timestamp,
        payload={
            "uuid": event_id,
            "event_type": "task_failed",
            "status": "failed",
            "timestamp": timestamp,
            "idempotency_key": "failure-analysis:1363",
        },
    )


def _make_runner(
    *,
    observe_source,
    agent_runner: AgentRunner,
    dispatcher,
    memory: _MemoryStore,
    config: OODAConfig | None = None,
    audit_writer=None,
    logger=None,
) -> OODARunner:
    return OODARunner(
        observe_source=observe_source,
        agent_runner=agent_runner,
        act_dispatcher=dispatcher,
        memory_append=memory.append,
        memory_read=memory.read_text,
        config=config or OODAConfig(observe_filter="failure-analysis"),
        audit_writer=audit_writer,
        logger=logger,
    )


def test_ac1_recover_task_flow(tmp_path):
    memory = _MemoryStore()
    dispatcher = ExecAppenderDispatcher(exec_jsonl_path=tmp_path / "exec.jsonl")
    agent = AgentRunner(
        llm_call=lambda prompt, tool_result=None: (
            '{"tool": "recover_task", "args": {"command": "claude -p", "uuid": "r1"}}'
        ),
        tool_executor=lambda tool, args: "",
        terminal_tools=frozenset({"recover_task", "escalate_to_slack", "skip"}),
    )
    runner = _make_runner(
        observe_source=_ListObserveSource([_failure_event()]),
        agent_runner=agent,
        dispatcher=dispatcher,
        memory=memory,
    )
    result = runner.run_tick()
    assert any(action.action == "recover_task" and action.success for action in result.actions_taken)


def test_ac2_feedback_visible_in_orient_prompt():
    memory = _MemoryStore()
    memory.append(
        role="system",
        content=json.dumps(
            {"event_id": "evt-1", "attempt": 1, "action": "recover_task", "success": False},
            ensure_ascii=False,
        ),
        timestamp="2026-05-29T10:05:00+09:00",
        source_tag="ooda_feedback",
        layer="ooda",
        dedupe_key="ooda:evt-1:1",
    )
    captured: list[str] = []

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        captured.append(prompt)
        return '{"tool": "skip", "args": {}}'

    runner = _make_runner(
        observe_source=_ListObserveSource([_failure_event()]),
        agent_runner=AgentRunner(
            llm_call=llm_call,
            tool_executor=lambda tool, args: "",
            terminal_tools=frozenset({"recover_task", "escalate_to_slack", "skip"}),
        ),
        dispatcher=_RecordingDispatcher(),
        memory=memory,
    )
    runner.run_tick()
    assert len(captured) == 1
    assert "evt-1" in captured[0]
    assert "recover_task" in captured[0]


def test_ac3_escalate_after_forces_slack_without_agent():
    memory = _MemoryStore()
    for attempt in (1, 2):
        memory.append(
            role="system",
            content=json.dumps(
                {"event_id": "evt-1", "attempt": attempt, "action": "recover_task", "success": False},
                ensure_ascii=False,
            ),
            timestamp=f"2026-05-29T10:0{attempt}:00+09:00",
            source_tag="ooda_feedback",
            layer="ooda",
            dedupe_key=f"ooda:evt-1:{attempt}",
        )

    agent_called = {"count": 0}

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        agent_called["count"] += 1
        return '{"tool": "recover_task", "args": {}}'

    dispatcher = _RecordingDispatcher()
    runner = _make_runner(
        observe_source=_ListObserveSource([_failure_event()]),
        agent_runner=AgentRunner(
            llm_call=llm_call,
            tool_executor=lambda tool, args: "",
            terminal_tools=frozenset({"recover_task", "escalate_to_slack", "skip"}),
        ),
        dispatcher=dispatcher,
        memory=memory,
        config=OODAConfig(escalate_after=2),
    )
    result = runner.run_tick()
    assert agent_called["count"] == 0
    assert result.escalated is True
    assert dispatcher.calls[0][0] == "escalate_to_slack"


def test_ac4_max_recovery_attempts_excludes_event():
    memory = _MemoryStore()
    for attempt in (1, 2, 3):
        memory.append(
            role="system",
            content=json.dumps(
                {"event_id": "evt-1", "attempt": attempt, "action": "recover_task", "success": False},
                ensure_ascii=False,
            ),
            timestamp=f"2026-05-29T10:0{attempt}:00+09:00",
            source_tag="ooda_feedback",
            layer="ooda",
            dedupe_key=f"ooda:evt-1:{attempt}",
        )
    dispatcher = _RecordingDispatcher()
    runner = _make_runner(
        observe_source=_ListObserveSource([_failure_event()]),
        agent_runner=AgentRunner(
            llm_call=lambda prompt, tool_result=None: '{"tool": "skip", "args": {}}',
            tool_executor=lambda tool, args: "",
            terminal_tools=frozenset({"recover_task", "escalate_to_slack", "skip"}),
        ),
        dispatcher=dispatcher,
        memory=memory,
        config=OODAConfig(max_recovery_attempts=3),
    )
    result = runner.run_tick()
    assert result.observed_events == 1
    assert result.actions_taken == []


def test_ac5_empty_observe_returns_zero_actions():
    memory = _MemoryStore()
    runner = _make_runner(
        observe_source=_ListObserveSource([]),
        agent_runner=AgentRunner(
            llm_call=lambda prompt, tool_result=None: None,
            tool_executor=lambda tool, args: "",
            terminal_tools=frozenset({"skip"}),
        ),
        dispatcher=_RecordingDispatcher(),
        memory=memory,
    )
    result = runner.run_tick()
    assert result.observed_events == 0
    assert result.actions_taken == []
    assert result.escalated is False


def test_ac6_agent_none_skips_event(caplog):
    memory = _MemoryStore()
    runner = _make_runner(
        observe_source=_ListObserveSource([_failure_event()]),
        agent_runner=AgentRunner(
            llm_call=lambda prompt, tool_result=None: None,
            tool_executor=lambda tool, args: "",
            terminal_tools=frozenset({"recover_task", "escalate_to_slack", "skip"}),
        ),
        dispatcher=_RecordingDispatcher(),
        memory=memory,
        logger=logging.getLogger("test.ooda"),
    )
    with caplog.at_level(logging.WARNING, logger="test.ooda"):
        result = runner.run_tick()
    assert result.actions_taken == []
    assert any("returned None" in record.message for record in caplog.records)


class _RaisingDispatcher:
    def dispatch(self, action: str, args: dict[str, Any]) -> ActResult:
        raise RuntimeError("dispatch failed")


def test_ac7_dispatch_exception_recorded_as_failure():
    memory = _MemoryStore()
    runner = _make_runner(
        observe_source=_ListObserveSource([_failure_event()]),
        agent_runner=AgentRunner(
            llm_call=lambda prompt, tool_result=None: (
                '{"tool": "recover_task", "args": {"command": "x"}}'
            ),
            tool_executor=lambda tool, args: "",
            terminal_tools=frozenset({"recover_task", "escalate_to_slack", "skip"}),
        ),
        dispatcher=_RaisingDispatcher(),
        memory=memory,
    )
    result = runner.run_tick()
    assert len(result.actions_taken) == 1
    assert result.actions_taken[0].success is False
    assert memory.entries
    assert memory.entries[-1]["dedupe_key"] == "ooda:evt-1:1"


def test_ac10_ooda_package_import_isolated():
  import importlib

  mod = importlib.import_module("mltgnt.ooda")
  assert mod.OODARunner is not None
  assert mod.OODAConfig is not None
  assert mod.OODATickResult is not None


def test_audit_integration_with_audit_jsonl(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(
        json.dumps(
            {
                "uuid": "fa-uuid",
                "event_type": "task_failed",
                "status": "failed",
                "timestamp": "2026-05-29T12:00:00+09:00",
                "idempotency_key": "failure-analysis:1363",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    memory = _MemoryStore()
    dispatcher = ExecAppenderDispatcher(exec_jsonl_path=tmp_path / "exec.jsonl")
    runner = _make_runner(
        observe_source=AuditJsonlSource(audit_path, observe_filter="failure-analysis"),
        agent_runner=AgentRunner(
            llm_call=lambda prompt, tool_result=None: (
                '{"tool": "recover_task", "args": {"command": "claude -p", "uuid": "r2"}}'
            ),
            tool_executor=lambda tool, args: "",
            terminal_tools=frozenset({"recover_task", "escalate_to_slack", "skip"}),
        ),
        dispatcher=dispatcher,
        memory=memory,
    )
    result = runner.run_tick()
    assert result.observed_events == 1
    assert result.actions_taken[0].action == "recover_task"
