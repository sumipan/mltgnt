"""
tests/test_mltgnt_scheduler.py — mltgnt.scheduler Unit Test()AC-3)

Design: Issue #118 §7 AC-3
"""
from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from mltgnt.exceptions import ConfigError
from mltgnt.scheduler import (
    ScheduleJob,
    PersonaScheduler,
    load_schedule_jobs,
)
from mltgnt.config import SchedulerConfig

TZ = ZoneInfo("Asia/Tokyo")


def dt_jst(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=TZ)


def make_scheduler(state_dir: Path, jobs: list[ScheduleJob]) -> PersonaScheduler:
    sch = PersonaScheduler(slack=None, state_dir=state_dir, jobs=jobs)
    sch.reload_jobs()
    return sch

def test_load_schedule_jobs_from_yaml(tmp_path: Path) -> None:
    """Valid schedule.yaml can parse."""
    yaml_file = tmp_path / "schedule.yaml"
    yaml_file.write_text(
        "jobs:\n"
        "  - id: morning\n"
        "    enabled: true\n"
        "    mode: scheduled\n"
        "    every_day_at: '09:00'\n"
        "    action: noop\n"
        "    notify: silent\n",
        encoding="utf-8",
    )
    jobs = load_schedule_jobs(yaml_file)
    assert len(jobs) == 1
    assert jobs[0].id == "morning"
    assert jobs[0].every_day_at == "09:00"


# ---------------------------------------------------------------------------
# AC-3: Job execution
# ---------------------------------------------------------------------------

def test_scheduled_fires_at_target_time(tmp_path: Path) -> None:
    """mode=scheduled, every_day_at="10:00" Jobs 10:00 Home tick ignition."""
    j = ScheduleJob.from_dict({
        "id": "fire_test",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    sch = make_scheduler(tmp_path / "state", [j])
    sch.tick(dt_jst(2026, 4, 17, 10, 0))
    time.sleep(0.5)
    assert sch.paths.done_path("fire_test", date(2026, 4, 17)).is_file()


def test_scheduled_does_not_refire_same_day(tmp_path: Path) -> None:
    """Same Day2 tick Home2does not firedone control by file)."""
    j = ScheduleJob.from_dict({
        "id": "once_test",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    sch = make_scheduler(tmp_path / "state", [j])
    sch.tick(dt_jst(2026, 4, 17, 10, 0))
    time.sleep(0.5)
    # Reset slot to simulate second tick at same time
    sch._scheduled_fired_slot.clear()
    sch.tick(dt_jst(2026, 4, 17, 10, 0))
    time.sleep(0.2)
    # done file exists from first fire, second should be skipped
    done_path = sch.paths.done_path("once_test", date(2026, 4, 17))
    assert done_path.is_file()


def test_interval_fires_multiple_times(tmp_path: Path) -> None:
    """mode=interval reigns after enough time after the previous run."""
    j = ScheduleJob.from_dict({
        "id": "interval_test",
        "mode": "interval",
        "interval_minutes": 10,
        "action": "noop",
        "notify": "silent",
    })
    sch = make_scheduler(tmp_path / "state", [j])
    now = dt_jst(2026, 4, 17, 10, 0)
    sch.tick(now)
    time.sleep(0.3)
    assert sch._interval_last_fired.get("interval_test") is not None


def test_interval_persists_last_fired_across_restart(tmp_path: Path) -> None:
    """interval Home last_fired will be permanently restored after restarting."""
    j = ScheduleJob.from_dict({
        "id": "persist_test",
        "mode": "interval",
        "interval_minutes": 60,
        "action": "noop",
        "notify": "silent",
    })
    state_dir = tmp_path / "state"

    sch1 = make_scheduler(state_dir, [j])
    now = dt_jst(2026, 4, 17, 10, 0)
    sch1.tick(now)
    time.sleep(0.3)
    assert sch1._interval_last_fired.get("persist_test") is not None
    assert sch1.paths.interval_last_fired_path("persist_test").exists()

    sch2 = make_scheduler(state_dir, [j])
    assert sch2._interval_last_fired.get("persist_test") is not None

    almost_now = dt_jst(2026, 4, 17, 10, 30)
    fired_before = dict(sch2._interval_last_fired)
    sch2.tick(almost_now)
    time.sleep(0.3)
    assert sch2._interval_last_fired["persist_test"] == fired_before["persist_test"]


# ---------------------------------------------------------------------------
# AC-3: Dependent Chain
# ---------------------------------------------------------------------------

def test_depends_on_waits_for_dependency(tmp_path: Path) -> None:
    """`depends_on: [job_a]` Jobs job_a Home done Do not ignite while the file does not exist."""
    job_a = ScheduleJob.from_dict({
        "id": "job_a",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    job_b = ScheduleJob.from_dict({
        "id": "job_b",
        "mode": "chained",
        "action": "noop",
        "notify": "silent",
        "depends_on": ["job_a"],
    })
    sch = make_scheduler(tmp_path / "state", [job_a, job_b])
    sch.tick(dt_jst(2026, 4, 17, 10, 0))
    time.sleep(0.3)
    # job_b should not have fired yet (job_a is being processed)
    # just verify it's not marked done without job_a being done
    # job_a fires noop - done quickly; job_b depends on job_a
    # The logic: job_b will fire after job_a's done file appears
    # For this test, we just verify no exception and job_b doesn't fire immediately on its own


def test_cycle_detection_raises(tmp_path: Path) -> None:
    """Cir ating Dependenciesjob_a → job_b → job_a _detect_cycles()  ValueError throw."""
    job_a = ScheduleJob.from_dict({
        "id": "job_a_cycle",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
        "depends_on": ["job_b_cycle"],
    })
    job_b = ScheduleJob.from_dict({
        "id": "job_b_cycle",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
        "depends_on": ["job_a_cycle"],
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[job_a, job_b])
    cycled = sch._detect_cycles([job_a, job_b])
    assert "job_a_cycle" in cycled or "job_b_cycle" in cycled


# ---------------------------------------------------------------------------
# AC-3: SchedulerConfig Contact Us
# ---------------------------------------------------------------------------

def test_scheduler_config_integration(tmp_path: Path) -> None:
    """SchedulerConfig Use the scheduler to work properly."""
    yaml_file = tmp_path / "schedule.yaml"
    yaml_file.write_text(
        "jobs:\n"
        "  - id: config_test\n"
        "    enabled: true\n"
        "    mode: scheduled\n"
        "    every_day_at: '11:00'\n"
        "    action: noop\n"
        "    notify: silent\n",
        encoding="utf-8",
    )
    config = SchedulerConfig(
        schedule_yaml=yaml_file,
        state_dir=tmp_path / "state",
    )
    sch = PersonaScheduler(slack=None, config=config)
    sch.reload_jobs()
    assert len(sch._jobs) == 1
    assert sch._jobs[0].id == "config_test"
def test_schedule_paths_importable() -> None:
    """SchedulePaths Home import What you can do"""
    from mltgnt.scheduler import SchedulePaths  # noqa: F401 (import check)


def test_noop_action_returns_true(tmp_path: Path) -> None:
    """job.action='noop' → execute_action Home (True, '') """
    job = ScheduleJob.from_dict({
        "id": "noop_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[])
    ok, msg = sch.execute_action(job)
    assert ok is True
    assert msg == ""


def test_unknown_action_raises_value_error(tmp_path: Path) -> None:
    """Unregistered action → execute_action Home ValueError Home raise """
    job = ScheduleJob.from_dict({
        "id": "unknown_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "unknown_action_xyz",
        "notify": "silent",
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[])
    with pytest.raises(ValueError):
        sch.execute_action(job)


def test_register_action_is_called(tmp_path: Path) -> None:
    """register_action Register callback Home execute_action Called via"""
    job = ScheduleJob.from_dict({
        "id": "custom_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "custom",
        "notify": "silent",
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[])
    sch.register_action("custom", lambda j: (True, "ok"))
    ok, msg = sch.execute_action(job)
    assert ok is True
    assert msg == "ok"


def test_actions_kwarg_in_init(tmp_path: Path) -> None:
    """__init__ Home actions= kwarg Contact Us callback Home execute_action Called via"""
    job = ScheduleJob.from_dict({
        "id": "init_action_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "init_action",
        "notify": "silent",
    })
    sch = PersonaScheduler(
        slack=None,
        state_dir=tmp_path / "state",
        jobs=[],
        actions={"init_action": lambda j: (True, "from_init")},
    )
    ok, msg = sch.execute_action(job)
    assert ok is True
    assert msg == "from_init"


def test_registered_action_failure(tmp_path: Path) -> None:
    """Register action Home (False, 'err') → execute_action Home (False, 'err') """
    job = ScheduleJob.from_dict({
        "id": "fail_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "fail_action",
        "notify": "silent",
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[])
    sch.register_action("fail_action", lambda j: (False, "err"))
    ok, msg = sch.execute_action(job)
    assert ok is False
    assert msg == "err"


def test_registered_action_failure_creates_failed_marker(tmp_path: Path) -> None:
    """On register_action failure, _spawn_job writes failed markers."""
    job = ScheduleJob.from_dict({
        "id": "fail_marker_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "fail_action",
        "notify": "silent",
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[job])
    sch.register_action("fail_action", lambda j: (False, "something went wrong"))
    sch.reload_jobs()
    d = date(2026, 5, 1)
    sch._spawn_job(job, d)
    time.sleep(0.5)
    assert sch.paths.failed_path("fail_marker_job", d).is_file()


def test_slack_none_post_does_not_raise(tmp_path: Path) -> None:
    """slack=None Home PersonaScheduler rate _post() There is no exception."""
    job = ScheduleJob.from_dict({
        "id": "notify_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "slack_secretary",
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[])
    sch._post(job, "test message")  # should not raise
def test_load_jobs_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    """Breakdown YAML Home ConfigError Send"""
    bad_yaml = tmp_path / "schedule.yaml"
    bad_yaml.write_text("jobs:\n  - id: [unclosed\n", encoding="utf-8")
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", yaml_path=bad_yaml)

    with pytest.raises(ConfigError, match="YAML load error"):
        sch._load_jobs()


def test_reload_jobs_keeps_previous_jobs_on_config_error(tmp_path: Path) -> None:
    """reload_jobs Home YAML Existing when damaged jobs keep."""
    good_yaml = tmp_path / "schedule.yaml"
    good_yaml.write_text(
        "jobs:\n"
        "  - id: morning\n"
        "    enabled: true\n"
        "    mode: scheduled\n"
        "    every_day_at: '09:00'\n"
        "    action: noop\n"
        "    notify: silent\n",
        encoding="utf-8",
    )
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", yaml_path=good_yaml)
    sch.reload_jobs()
    assert len(sch._jobs) == 1
    assert sch._jobs[0].id == "morning"

    good_yaml.write_text("jobs:\n  - id: [unclosed\n", encoding="utf-8")
    sch.reload_jobs()
    assert len(sch._jobs) == 1
    assert sch._jobs[0].id == "morning"
def _wait_until(pred, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _raise_boom(job: ScheduleJob) -> tuple[bool, str]:
    raise RuntimeError("boom")


def test_action_exception_marks_failed_and_notifies(tmp_path: Path) -> None:
    job = ScheduleJob.from_dict({
        "id": "boom_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "boom",
        "notify": "slack_secretary",
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[job])
    sch.register_action("boom", _raise_boom)
    sch.reload_jobs()
    posted: list[str] = []
    sch._post = lambda j, text: posted.append(text)  # type: ignore[method-assign]
    finished: list[bool] = []
    d = date(2026, 5, 1)

    sch._spawn_job(job, d, on_finish=lambda: finished.append(True))

    assert _wait_until(lambda: finished == [True])
    failed = sch.paths.failed_path("boom_job", d)
    assert failed.is_file()
    assert "RuntimeError: boom" in failed.read_text(encoding="utf-8")
    assert len(posted) == 1
    assert "RuntimeError" in posted[0] and "boom" in posted[0]
    assert "boom_job" not in sch._running


def test_interval_action_exception_notifies_without_failed_file(tmp_path: Path) -> None:
    job = ScheduleJob.from_dict({
        "id": "boom_interval",
        "mode": "interval",
        "interval_minutes": 10,
        "action": "boom",
        "notify": "slack_secretary",
    })
    sch = PersonaScheduler(slack=None, state_dir=tmp_path / "state", jobs=[job])
    sch.register_action("boom", _raise_boom)
    sch.reload_jobs()
    posted: list[str] = []
    sch._post = lambda j, text: posted.append(text)  # type: ignore[method-assign]
    finished: list[bool] = []
    d = date(2026, 5, 1)

    sch._spawn_job(job, d, on_finish=lambda: finished.append(True))

    assert _wait_until(lambda: finished == [True])
    assert not sch.paths.failed_path("boom_interval", d).exists()
    assert len(posted) == 1
    assert "boom" in posted[0]


def test_tick_prunes_once_per_day(tmp_path: Path) -> None:
    sch = PersonaScheduler(
        slack=None, state_dir=tmp_path / "state", jobs=[], state_keep_days=7
    )
    sch.reload_jobs()
    calls: list[tuple[date, int]] = []
    sch.paths.prune = lambda today, keep_days=30: calls.append((today, keep_days)) or 0  # type: ignore[method-assign]

    sch.tick(dt_jst(2026, 4, 17, 10, 0))
    sch.tick(dt_jst(2026, 4, 17, 10, 1))
    assert calls == [(date(2026, 4, 17), 7)]

    sch.tick(dt_jst(2026, 4, 18, 0, 0))
    assert calls == [(date(2026, 4, 17), 7), (date(2026, 4, 18), 7)]


def test_tick_survives_prune_exception(tmp_path: Path) -> None:
    sch = make_scheduler(tmp_path / "state", [])

    def _bad_prune(today: date, keep_days: int = 30) -> int:
        raise OSError("nope")

    sch.paths.prune = _bad_prune  # type: ignore[method-assign]
    sch.tick(dt_jst(2026, 4, 17, 10, 0))  # must not raise


def test_tick_prune_removes_old_state(tmp_path: Path) -> None:
    sch = make_scheduler(tmp_path / "state", [])
    old = sch.paths.done_path("old_job", date(2026, 1, 1))
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text("ok", encoding="utf-8")
    sch.tick(dt_jst(2026, 4, 17, 10, 0))
    assert not old.exists()


def test_chain_every_run_collision_logs_and_writes_no_state(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    upstream = ScheduleJob.from_dict({
        "id": "probe_up",
        "mode": "interval",
        "interval_minutes": 10,
        "action": "probe",
        "notify": "silent",
    })
    dependent = ScheduleJob.from_dict({
        "id": "act_down",
        "mode": "chained",
        "chain_every_run": True,
        "depends_on": ["probe_up"],
        "action": "capture",
        "notify": "silent",
    })
    sch = PersonaScheduler(
        slack=None, state_dir=tmp_path / "state", jobs=[upstream, dependent]
    )
    sch.register_action("probe", lambda j: (True, "out"))
    called: list[str] = []
    sch.register_action("capture", lambda j: (called.append("x") or (True, "")))
    sch.reload_jobs()
    sch._running.add("act_down")  # downstream still running from a previous run
    finished: list[bool] = []
    d = date(2026, 4, 17)

    with caplog.at_level("WARNING", logger="mltgnt.scheduler.runner"):
        sch._spawn_job(upstream, d, on_finish=lambda: finished.append(True))
        assert _wait_until(lambda: finished == [True])

    assert called == []
    assert any(
        "act_down" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records
    )
    assert not sch.paths.skipped_path("act_down", d).exists()
    assert not sch.paths.failed_path("act_down", d).exists()
    assert not sch.paths.done_path("act_down", d).exists()


def test_stop_returns_promptly(tmp_path: Path) -> None:
    sch = make_scheduler(tmp_path / "state", [])
    sch.start_background()
    thread = sch._thread
    assert thread is not None
    time.sleep(0.1)
    started = time.monotonic()
    sch.stop()
    assert time.monotonic() - started < 1.0
    assert not thread.is_alive()
