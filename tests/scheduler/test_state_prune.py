"""SchedulePaths.prune: drop dated state files older than keep_days (#5034)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from mltgnt.scheduler.state import SchedulePaths

TODAY = date(2026, 10, 9)


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")
    return p


def test_prune_removes_files_older_than_keep_days(tmp_path: Path) -> None:
    paths = SchedulePaths(tmp_path / "state")
    old = TODAY - timedelta(days=31)
    boundary = TODAY - timedelta(days=30)
    recent = TODAY - timedelta(days=29)

    old_files = [
        _touch(paths.done_path("job_a", old)),
        _touch(paths.planned_path("job_a", old)),
        _touch(paths.missed_path("job_a", old)),
        _touch(paths.failed_path("job_a", old)),
        _touch(paths.skipped_path("job_a", old)),
    ]
    kept = [
        _touch(paths.done_path("job_a", boundary)),
        _touch(paths.done_path("job_a", recent)),
        _touch(paths.failed_path("job_with_under_score", recent)),
        _touch(paths.interval_last_fired_path("job_a")),
    ]
    paths.write_interval_last_fired("job_b", datetime(2020, 1, 1))
    kept.append(paths.interval_last_fired_path("job_b"))

    removed = paths.prune(TODAY, keep_days=30)

    assert removed == 5
    for p in old_files:
        assert not p.exists()
    for p in kept:
        assert p.exists()


def test_prune_handles_job_ids_with_underscores(tmp_path: Path) -> None:
    paths = SchedulePaths(tmp_path / "state")
    p = _touch(paths.done_path("my_daily_job", TODAY - timedelta(days=40)))
    assert paths.prune(TODAY) == 1
    assert not p.exists()


def test_prune_keeps_unparseable_names(tmp_path: Path) -> None:
    paths = SchedulePaths(tmp_path / "state")
    tmp = _touch(paths.done_dir / "foo.tmp")
    nodate = _touch(paths.done_dir / "job_notadate.done")
    assert paths.prune(TODAY) == 0
    assert tmp.exists()
    assert nodate.exists()


def test_prune_missing_dirs_returns_zero(tmp_path: Path) -> None:
    paths = SchedulePaths(tmp_path / "missing")
    assert paths.prune(TODAY) == 0


@pytest.mark.parametrize("keep_days", [0, -1])
def test_prune_rejects_keep_days_below_one(tmp_path: Path, keep_days: int) -> None:
    paths = SchedulePaths(tmp_path / "state")
    with pytest.raises(ValueError):
        paths.prune(TODAY, keep_days=keep_days)
