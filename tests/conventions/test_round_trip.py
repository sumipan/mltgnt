"""Round trips on real files: memory, scheduler state and the DagHooks audit adapter."""
from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from ghdag.dag.hooks import DagHooks, Task, TaskMetrics

from mltgnt.bridges.hooks_adapter import MltgntHooks
from mltgnt.config import MemoryConfig
from mltgnt.memory import append_memory_entry, compact, memory_file_path, parse_jsonl
from mltgnt.scheduler.state import SchedulePaths


def _identity_llm(prompt: str) -> str:
    return prompt


def test_memory_entries_survive_compaction(tmp_path: Path) -> None:
    config = MemoryConfig(chat_dir=tmp_path)
    assert append_memory_entry(config, "p", "user", "first entry", "2026-09-20T10:00:00+09:00", source_tag="chat")
    assert append_memory_entry(
        config, "p", "assistant", "second entry", "2026-09-21T10:00:00+09:00", source_tag="chat"
    )
    path = memory_file_path(config, "p")
    assert [e.content for e in parse_jsonl(path)] == ["first entry", "second entry"]

    compact(config, "p", llm_call=_identity_llm)

    assert [e.content for e in parse_jsonl(path)] == ["first entry", "second entry"]


@pytest.mark.parametrize(
    "dt",
    [
        datetime(2026, 9, 25, 12, 30, 15),
        datetime(2026, 9, 25, 12, 30, 15, tzinfo=timezone(timedelta(hours=9))),
    ],
    ids=["naive", "aware"],
)
def test_interval_last_fired_round_trip(tmp_path: Path, dt: datetime) -> None:
    paths = SchedulePaths(tmp_path)
    paths.write_interval_last_fired("job-a", dt)
    loaded = paths.load_all_interval_last_fired()
    assert loaded == {"job-a": dt}
    assert loaded["job-a"].tzinfo == dt.tzinfo


def test_on_task_success_writes_audit(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    hooks = MltgntHooks(audit)
    task = Task(uuid="task-uuid-1", command="echo ok", model="m1")
    metrics = TaskMetrics(
        uuid="task-uuid-1",
        engine="claude",
        model="m1",
        wall_time_sec=1.5,
        token_count=10,
        status="success",
        started_at=0.0,
        finished_at=1.5,
    )
    hooks.on_task_success("task-uuid-1", task, metrics)

    last = json.loads(audit.read_text(encoding="utf-8").splitlines()[-1])
    assert last["event_type"] == "task_success"
    assert last["uuid"] == "task-uuid-1"


def _public_methods(cls: type) -> list[str]:
    return sorted(n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n)))


def _param_names(func: object) -> list[str]:
    return [name for name in inspect.signature(func).parameters if name != "self"]  # type: ignore[arg-type]


def test_mltgnt_hooks_matches_dag_hooks_signatures() -> None:
    methods = _public_methods(DagHooks)
    assert len(methods) >= 12
    for name in methods:
        assert hasattr(MltgntHooks, name), name
        assert _param_names(getattr(MltgntHooks, name)) == _param_names(getattr(DagHooks, name)), name
