"""
mltgnt.scheduler — job dispatch core + YAML parser.

Origin: core + YAML parser from tools/secretary/scheduler.py
Takes SchedulerConfig. Persona-related callbacks are injected via __init__.

Design: Issue #118 §3 (T4)
"""
from __future__ import annotations

from mltgnt.scheduler.fanout import _FANOUT_PROMPT_SUFFIX, _parse_fanout_steps  # noqa: F401
from mltgnt.scheduler.loader import load_schedule_jobs
from mltgnt.scheduler.models import (  # noqa: F401
    ActionFn,
    DAY_NAMES,
    ScheduleJob,
    _DEFAULT_TIMEZONE,
    _parse_hhmm,
    _to_minutes_since_midnight,
)
from mltgnt.scheduler.runner import PersonaScheduler
from mltgnt.scheduler.state import SchedulePaths, _hash_offset, atomic_write_text

__all__ = [
    "ScheduleJob",
    "PersonaScheduler",
    "SchedulePaths",
    "load_schedule_jobs",
    "atomic_write_text",
    "_hash_offset",
]
