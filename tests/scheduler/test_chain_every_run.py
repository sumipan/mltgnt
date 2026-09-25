"""chain_every_run: chained jobs that fire after every successful upstream run."""
from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from mltgnt.scheduler import PersonaScheduler, ScheduleJob

TZ = ZoneInfo("Asia/Tokyo")


def _dt(hour: int, minute: int) -> datetime:
    return datetime(2026, 4, 17, hour, minute, tzinfo=TZ)


def _wait_until(pred, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.02)


def _upstream(job_id: str = "check") -> ScheduleJob:
    return ScheduleJob.from_dict({
        "id": job_id,
        "mode": "interval",
        "interval_minutes": 10,
        "action": "probe",
        "notify": "silent",
        "on_exit": {"nonzero": "skip"},
    })


def _dependent(job_id: str = "act", depends_on: str = "check") -> ScheduleJob:
    return ScheduleJob.from_dict({
        "id": job_id,
        "mode": "chained",
        "chain_every_run": True,
        "depends_on": [depends_on],
        "action": "capture",
        "notify": "silent",
    })


def test_from_dict_parses_chain_every_run() -> None:
    job = _dependent()
    assert job.chain_every_run is True
    assert job.upstream_output is None


def test_chain_every_run_requires_chained_mode_and_depends_on() -> None:
    with pytest.raises(ValueError, match="mode: chained"):
        ScheduleJob.from_dict({
            "id": "bad_mode",
            "mode": "interval",
            "interval_minutes": 5,
            "chain_every_run": True,
            "depends_on": ["x"],
            "action": "noop",
            "notify": "silent",
        })
    with pytest.raises(ValueError, match="depends_on"):
        ScheduleJob.from_dict({
            "id": "bad_deps",
            "mode": "chained",
            "chain_every_run": True,
            "action": "noop",
            "notify": "silent",
        })


def test_dependent_fires_with_upstream_output_and_writes_no_marks(tmp_path: Path) -> None:
    captured: list[str | None] = []
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[_upstream(), _dependent()])
    sch.register_action("probe", lambda job: (True, "free_slots=2"))
    sch.register_action("capture", lambda job: (captured.append(job.upstream_output) or (True, "done")))
    sch.reload_jobs()

    sch.tick(_dt(10, 0))
    _wait_until(lambda: len(captured) == 1)

    assert captured == ["free_slots=2"]
    d = date(2026, 4, 17)
    assert not sch.paths.done_path("act", d).is_file()
    assert not sch.paths.done_path("check", d).is_file()


def test_dependent_fires_again_on_next_upstream_run(tmp_path: Path) -> None:
    captured: list[str | None] = []
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[_upstream(), _dependent()])
    sch.register_action("probe", lambda job: (True, "tick"))
    sch.register_action("capture", lambda job: (captured.append(job.upstream_output) or (True, "")))
    sch.reload_jobs()

    sch.tick(_dt(10, 0))
    _wait_until(lambda: len(captured) == 1)
    sch.tick(_dt(10, 15))
    _wait_until(lambda: len(captured) == 2)

    assert captured == ["tick", "tick"]


def test_dependent_is_never_time_triggered(tmp_path: Path) -> None:
    captured: list[str | None] = []
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[_dependent()])
    sch.register_action("capture", lambda job: (captured.append(job.upstream_output) or (True, "")))
    sch.reload_jobs()

    for minute in range(0, 60, 5):
        sch.tick(_dt(10, minute))
    time.sleep(0.2)

    assert captured == []


def test_dependent_not_fired_when_upstream_skips_or_fails(tmp_path: Path) -> None:
    captured: list[str | None] = []
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[_upstream(), _dependent()])
    sch.register_action("probe", lambda job: (False, "nothing to do"))
    sch.register_action("capture", lambda job: (captured.append(job.upstream_output) or (True, "")))
    sch.reload_jobs()

    sch.tick(_dt(10, 0))
    time.sleep(0.3)

    assert captured == []


def test_original_job_object_is_not_mutated(tmp_path: Path) -> None:
    dep = _dependent()
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[_upstream(), dep])
    sch.register_action("probe", lambda job: (True, "payload"))
    seen: list[str | None] = []
    sch.register_action("capture", lambda job: (seen.append(job.upstream_output) or (True, "")))
    sch.reload_jobs()

    sch.tick(_dt(10, 0))
    _wait_until(lambda: len(seen) == 1)

    assert seen == ["payload"]
    assert dep.upstream_output is None
