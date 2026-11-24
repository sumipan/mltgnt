"""chain_every_run: chained jobs that fire after every successful upstream run."""
from __future__ import annotations

import threading
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from mltgnt.scheduler import PersonaScheduler, ScheduleJob

TZ = ZoneInfo("Asia/Tokyo")


def _dt(hour: int, minute: int) -> datetime:
    return datetime(2026, 4, 17, hour, minute, tzinfo=TZ)


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
    finished = threading.Event()
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[_upstream(), _dependent()])
    sch.register_action("probe", lambda job: (True, "free_slots=2"))

    def capture(job: ScheduleJob) -> tuple[bool, str]:
        captured.append(job.upstream_output)
        finished.set()
        return True, "done"

    sch.register_action("capture", capture)
    sch.reload_jobs()

    sch.tick(_dt(10, 0))
    assert finished.wait()

    assert captured == ["free_slots=2"]
    d = date(2026, 4, 17)
    assert not sch.paths.done_path("act", d).is_file()
    assert not sch.paths.done_path("check", d).is_file()


def test_dependent_fires_again_on_next_upstream_run(tmp_path: Path) -> None:
    captured: list[str | None] = []
    upstream_finished = [threading.Event(), threading.Event()]
    dependent_finished = [threading.Event(), threading.Event()]
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[_upstream(), _dependent()])
    sch.register_action("probe", lambda job: (True, "tick"))

    def capture(job: ScheduleJob) -> tuple[bool, str]:
        captured.append(job.upstream_output)
        return True, ""

    sch.register_action("capture", capture)
    sch.reload_jobs()

    original_spawn = sch._spawn_job
    run_index = {"check": 0, "act": 0}

    def spawn_with_completion(
        job: ScheduleJob,
        d: date,
        on_finish=None,
        upstream_output: str | None = None,
    ) -> None:
        index = run_index[job.id]
        run_index[job.id] += 1
        completion = upstream_finished[index] if job.id == "check" else dependent_finished[index]
        original_spawn(job, d, on_finish=completion.set, upstream_output=upstream_output)

    sch._spawn_job = spawn_with_completion  # type: ignore[method-assign]

    sch.tick(_dt(10, 0))
    assert upstream_finished[0].wait()
    assert dependent_finished[0].wait()
    sch.tick(_dt(10, 15))
    assert upstream_finished[1].wait()
    assert dependent_finished[1].wait()

    assert captured == ["tick", "tick"]


def test_dependent_is_never_time_triggered(tmp_path: Path) -> None:
    captured: list[str | None] = []
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[_dependent()])
    sch.register_action("capture", lambda job: (captured.append(job.upstream_output) or (True, "")))
    sch.reload_jobs()

    for minute in range(0, 60, 5):
        sch.tick(_dt(10, minute))

    assert captured == []


def test_dependent_not_fired_when_upstream_skips_or_fails(tmp_path: Path) -> None:
    captured: list[str | None] = []
    upstream_finished = threading.Event()
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[_upstream(), _dependent()])
    sch.register_action("probe", lambda job: (False, "nothing to do"))
    sch.register_action("capture", lambda job: (captured.append(job.upstream_output) or (True, "")))
    sch.reload_jobs()

    original_spawn = sch._spawn_job

    def spawn_with_completion(job, d, on_finish=None, upstream_output=None):
        original_spawn(job, d, on_finish=upstream_finished.set, upstream_output=upstream_output)

    sch._spawn_job = spawn_with_completion  # type: ignore[method-assign]

    sch.tick(_dt(10, 0))
    assert upstream_finished.wait()

    assert captured == []


def test_original_job_object_is_not_mutated(tmp_path: Path) -> None:
    dep = _dependent()
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[_upstream(), dep])
    sch.register_action("probe", lambda job: (True, "payload"))
    seen: list[str | None] = []
    finished = threading.Event()

    def capture(job: ScheduleJob) -> tuple[bool, str]:
        seen.append(job.upstream_output)
        finished.set()
        return True, ""

    sch.register_action("capture", capture)
    sch.reload_jobs()

    sch.tick(_dt(10, 0))
    assert finished.wait()

    assert seen == ["payload"]
    assert dep.upstream_output is None
