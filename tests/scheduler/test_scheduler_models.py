"""ScheduleJob model and on_exit policy tests."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from mltgnt.scheduler import ScheduleJob
from mltgnt.scheduler.models import OnExitPolicy
from mltgnt.scheduler.state import SchedulePaths as _SchedulePaths

def test_from_dict_valid_scheduled() -> None:
    """Valid scheduled Job fields are set correctly."""
    job = ScheduleJob.from_dict({
        "id": "test_job",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    assert job.id == "test_job"
    assert job.mode == "scheduled"
    assert job.every_day_at == "10:00"
    assert job.action == "noop"
    assert job.notify == "silent"
    assert job.enabled is True


def test_from_dict_invalid_mode_raises() -> None:
    """`mode` outside scheduled|fuzzy_window|interval|chained → ValueError."""
    with pytest.raises(ValueError, match="unknown mode"):
        ScheduleJob.from_dict({
            "id": "bad",
            "mode": "unknown_mode",
            "action": "noop",
            "notify": "silent",
        })


def test_from_dict_invalid_hhmm_raises() -> None:
    """`every_day_at` not HH:MM → ValueError."""
    with pytest.raises(ValueError):
        ScheduleJob.from_dict({
            "id": "bad_time",
            "mode": "scheduled",
            "every_day_at": "25:99",
            "action": "noop",
            "notify": "silent",
        })


def test_overnight_fuzzy_window_raises() -> None:
    """Overnight fuzzy window → ValueError."""
    with pytest.raises(ValueError, match="overnight"):
        ScheduleJob.from_dict({
            "id": "overnight",
            "mode": "fuzzy_window",
            "window_start": "23:00",
            "window_end": "01:00",
            "action": "noop",
            "notify": "silent",
        })
def test_from_dict_on_exit_skip() -> None:
    """`on_exit: {nonzero: skip}` → `OnExitPolicy(nonzero="skip")`."""
    job = ScheduleJob.from_dict({
        "id": "j",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
        "on_exit": {"nonzero": "skip"},
    })
    assert job.on_exit == OnExitPolicy(nonzero="skip")


def test_from_dict_on_exit_fail_explicit() -> None:
    """`on_exit: {nonzero: fail}` → `OnExitPolicy(nonzero="fail")`."""
    job = ScheduleJob.from_dict({
        "id": "j",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
        "on_exit": {"nonzero": "fail"},
    })
    assert job.on_exit == OnExitPolicy(nonzero="fail")


def test_from_dict_on_exit_none() -> None:
    """`on_exit` Not specified → `job.on_exit is None`."""
    job = ScheduleJob.from_dict({
        "id": "j",
        "mode": "scheduled",
        "every_day_at": "10:00",
        "action": "noop",
        "notify": "silent",
    })
    assert job.on_exit is None


def test_from_dict_on_exit_invalid_value() -> None:
    """`on_exit: {nonzero: retry}` → `ValueError`."""
    with pytest.raises(ValueError, match="fail.*skip|skip.*fail"):
        ScheduleJob.from_dict({
            "id": "j",
            "mode": "scheduled",
            "every_day_at": "10:00",
            "action": "noop",
            "notify": "silent",
            "on_exit": {"nonzero": "retry"},
        })


def test_from_dict_on_exit_empty_dict() -> None:
    """`on_exit: {}` (English) → `ValueError`."""
    with pytest.raises(ValueError, match="nonzero"):
        ScheduleJob.from_dict({
            "id": "j",
            "mode": "scheduled",
            "every_day_at": "10:00",
            "action": "noop",
            "notify": "silent",
            "on_exit": {},
        })


def test_from_dict_on_exit_not_dict() -> None:
    """`on_exit: "skip"` () → `ValueError`."""
    with pytest.raises(ValueError, match="dict"):
        ScheduleJob.from_dict({
            "id": "j",
            "mode": "scheduled",
            "every_day_at": "10:00",
            "action": "noop",
            "notify": "silent",
            "on_exit": "skip",
        })


def test_skipped_path_format(tmp_path: Path) -> None:
    """`SchedulePaths.skipped_path("j1", date(2026,1,1))` → `<state_dir>/skipped/j1_2026-01-01.skipped`."""
    p = _SchedulePaths(tmp_path / "state")
    result = p.skipped_path("j1", date(2026, 1, 1))
    assert result == tmp_path / "state" / "skipped" / "j1_2026-01-01.skipped"
