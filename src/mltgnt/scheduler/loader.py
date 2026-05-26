from __future__ import annotations

from pathlib import Path

import yaml

from mltgnt.scheduler.models import ScheduleJob, _DEFAULT_TIMEZONE


def load_schedule_jobs(
    yaml_path: Path,
    *,
    default_timezone: str = _DEFAULT_TIMEZONE,
) -> list[ScheduleJob]:
    if not yaml_path.is_file():
        return []
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    jobs_raw = data.get("jobs") or []
    if not isinstance(jobs_raw, list):
        raise ValueError("schedule.yaml: jobs はリストである必要があります")
    return [ScheduleJob.from_dict(j, default_timezone=default_timezone) for j in jobs_raw]
