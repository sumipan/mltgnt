"""
mltgnt.scheduler — ジョブディスパッチコア + YAML パーサ。

元コード: tools/secretary/scheduler.py のコア + YAML パーサ
SchedulerConfig 引数で受け取る。ペルソナ関連コールバックは __init__ 引数で注入。

設計: Issue #118 §3 (T4)
"""
from __future__ import annotations

from mltgnt.scheduler.fanout import _FANOUT_PROMPT_SUFFIX  # noqa: F401
from mltgnt.scheduler.loader import load_schedule_jobs
from mltgnt.scheduler.models import ScheduleJob
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
