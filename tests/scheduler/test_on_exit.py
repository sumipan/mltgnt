"""
tests/scheduler/test_on_exit.py — unit tests for on_exit policy and skip propagation (Issue #2380)

7 cases:
  1. on_exit unset + ok=True  → only done_path is written
  2. on_exit unset + ok=False → only failed_path is written
  3. on_exit.nonzero=skip + ok=False → only skipped_path; _post/_record_to_memory not called
  4. on_exit.nonzero=fail + ok=False → only failed_path (same as unset)
  5. chained dependency skipped → self also skipped_path; execute_action not called
  6. scheduled dependency skipped → self also skipped_path; execute_action not called
  7. fuzzy_window dependency skipped → self also skipped_path; execute_action not called
"""
from __future__ import annotations

import threading
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from mltgnt.scheduler.models import OnExitPolicy, ScheduleJob
from mltgnt.scheduler.runner import PersonaScheduler
from mltgnt.scheduler.state import SchedulePaths

TZ = ZoneInfo("Asia/Tokyo")


def dt_jst(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=TZ)


def make_scheduler(state_dir: Path, jobs: list[ScheduleJob]) -> PersonaScheduler:
    sch = PersonaScheduler(slack=None, state_dir=state_dir, jobs=jobs)
    sch.reload_jobs()
    return sch


def wait_idle(sch: PersonaScheduler, timeout: float = 2.0) -> None:
    """Wait until all job threads have finished."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with sch._run_lock:
            if not sch._running:
                return
        time.sleep(0.05)
    raise TimeoutError("_running did not become empty")


# ---------------------------------------------------------------------------
# Case 1: on_exit unset + ok=True → only done_path is written
# ---------------------------------------------------------------------------

def test_no_on_exit_ok_true_writes_done(tmp_path: Path) -> None:
    job = ScheduleJob.from_dict({
        "id": "job_a",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    sch = make_scheduler(tmp_path, [job])
    d = date(2026, 7, 21)

    with patch.object(sch, "execute_action", return_value=(True, "ok")) as mock_exec:
        sch._spawn_job(job, d)
        wait_idle(sch)

    mock_exec.assert_called_once()
    assert sch.paths.done_path(job.id, d).is_file()
    assert not sch.paths.failed_path(job.id, d).is_file()
    assert not sch.paths.skipped_path(job.id, d).is_file()


# ---------------------------------------------------------------------------
# Case 2: on_exit unset + ok=False → only failed_path is written
# ---------------------------------------------------------------------------

def test_no_on_exit_ok_false_writes_failed(tmp_path: Path) -> None:
    job = ScheduleJob.from_dict({
        "id": "job_b",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    sch = make_scheduler(tmp_path, [job])
    d = date(2026, 7, 21)

    with patch.object(sch, "execute_action", return_value=(False, "error msg")):
        sch._spawn_job(job, d)
        wait_idle(sch)

    assert not sch.paths.done_path(job.id, d).is_file()
    assert sch.paths.failed_path(job.id, d).is_file()
    assert not sch.paths.skipped_path(job.id, d).is_file()


# ---------------------------------------------------------------------------
# Case 3: on_exit.nonzero=skip + ok=False → only skipped_path is written
#            _post / _record_to_memory are not called
# ---------------------------------------------------------------------------

def test_on_exit_skip_ok_false_writes_skipped(tmp_path: Path) -> None:
    job = ScheduleJob.from_dict({
        "id": "job_c",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
        "on_exit": {"nonzero": "skip"},
    })
    sch = make_scheduler(tmp_path, [job])
    d = date(2026, 7, 21)

    with (
        patch.object(sch, "execute_action", return_value=(False, "error")) as mock_exec,
        patch.object(sch, "_post") as mock_post,
        patch.object(sch, "_record_to_memory") as mock_mem,
    ):
        sch._spawn_job(job, d)
        wait_idle(sch)

    mock_exec.assert_called_once()
    assert not sch.paths.done_path(job.id, d).is_file()
    assert not sch.paths.failed_path(job.id, d).is_file()
    assert sch.paths.skipped_path(job.id, d).is_file()
    mock_post.assert_not_called()
    mock_mem.assert_not_called()


# ---------------------------------------------------------------------------
# Case 4: on_exit.nonzero=fail + ok=False → only failed_path (same as unset)
# ---------------------------------------------------------------------------

def test_on_exit_fail_ok_false_writes_failed(tmp_path: Path) -> None:
    job = ScheduleJob.from_dict({
        "id": "job_d",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
        "on_exit": {"nonzero": "fail"},
    })
    sch = make_scheduler(tmp_path, [job])
    d = date(2026, 7, 21)

    with patch.object(sch, "execute_action", return_value=(False, "err")):
        sch._spawn_job(job, d)
        wait_idle(sch)

    assert not sch.paths.done_path(job.id, d).is_file()
    assert sch.paths.failed_path(job.id, d).is_file()
    assert not sch.paths.skipped_path(job.id, d).is_file()


# ---------------------------------------------------------------------------
# Case 5: chained dependency skipped → self also skipped_path; execute_action not called
# ---------------------------------------------------------------------------

def test_chained_dep_skipped_propagates_skip(tmp_path: Path) -> None:
    dep = ScheduleJob.from_dict({
        "id": "dep_job",
        "mode": "scheduled",
        "every_day_at": "09:00",
        "action": "noop",
        "notify": "silent",
    })
    child = ScheduleJob.from_dict({
        "id": "child_job",
        "mode": "chained",
        "action": "noop",
        "notify": "silent",
        "depends_on": ["dep_job"],
    })
    sch = make_scheduler(tmp_path, [dep, child])
    d = date(2026, 7, 21)

    # Mark dependency as skipped
    sch._mark_skipped(dep, d)

    now = dt_jst(2026, 7, 21, 10, 0)
    with patch.object(sch, "execute_action") as mock_exec:
        sch.tick(now)

    mock_exec.assert_not_called()
    assert sch.paths.skipped_path(child.id, d).is_file()
    assert not sch.paths.done_path(child.id, d).is_file()
    assert not sch.paths.failed_path(child.id, d).is_file()


# ---------------------------------------------------------------------------
# Case 6: scheduled dependency skipped → self also skipped_path; execute_action not called
# ---------------------------------------------------------------------------

def test_scheduled_dep_skipped_propagates_skip(tmp_path: Path) -> None:
    dep = ScheduleJob.from_dict({
        "id": "dep_sched",
        "mode": "scheduled",
        "every_day_at": "09:00",
        "action": "noop",
        "notify": "silent",
    })
    child = ScheduleJob.from_dict({
        "id": "child_sched",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
        "depends_on": ["dep_sched"],
    })
    sch = make_scheduler(tmp_path, [dep, child])
    d = date(2026, 7, 21)

    sch._mark_skipped(dep, d)

    now = dt_jst(2026, 7, 21, 10, 0)
    with patch.object(sch, "execute_action") as mock_exec:
        sch.tick(now)

    mock_exec.assert_not_called()
    assert sch.paths.skipped_path(child.id, d).is_file()
    assert not sch.paths.done_path(child.id, d).is_file()
    assert not sch.paths.failed_path(child.id, d).is_file()


# ---------------------------------------------------------------------------
# Case 7: fuzzy_window dependency skipped → self also skipped_path; execute_action not called
# ---------------------------------------------------------------------------

def test_fuzzy_window_dep_skipped_propagates_skip(tmp_path: Path) -> None:
    dep = ScheduleJob.from_dict({
        "id": "dep_fuzzy",
        "mode": "scheduled",
        "every_day_at": "08:00",
        "action": "noop",
        "notify": "silent",
    })
    child = ScheduleJob.from_dict({
        "id": "child_fuzzy",
        "mode": "fuzzy_window",
        "window_start": "09:00",
        "window_end": "11:00",
        "fuzzy_method": "hash",
        "action": "noop",
        "notify": "silent",
        "depends_on": ["dep_fuzzy"],
    })
    sch = make_scheduler(tmp_path, [dep, child])
    d = date(2026, 7, 21)

    sch._mark_skipped(dep, d)

    # Write planned so the hash-chosen run_minute comes before 10:00
    sch._write_planned_minute(child, d, 9 * 60)  # 09:00

    now = dt_jst(2026, 7, 21, 10, 0)
    with patch.object(sch, "execute_action") as mock_exec:
        sch.tick(now)

    mock_exec.assert_not_called()
    assert sch.paths.skipped_path(child.id, d).is_file()
    assert not sch.paths.done_path(child.id, d).is_file()
    assert not sch.paths.failed_path(child.id, d).is_file()
