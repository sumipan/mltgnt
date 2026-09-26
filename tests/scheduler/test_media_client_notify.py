"""PersonaScheduler notifies through MediaClient and the legacy protocol (#4027)."""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pytest

from mltgnt.interfaces.media import Status
from mltgnt.scheduler import PersonaScheduler
from mltgnt.scheduler.models import ScheduleJob


class FakeMedia:
    def __init__(self) -> None:
        self.posts: list[tuple[str, str, str | None]] = []

    def post(self, text: str, space: str, thread: str | None = None) -> str | None:
        self.posts.append((text, space, thread))
        return "M1"

    def update(self, message_id: str, text: str) -> bool:
        return True

    def set_status(self, message_id: str, status: Status) -> bool:
        return True

    def upload(self, path: str, space: str, thread: str | None = None) -> bool:
        return False


class LegacySlack:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []

    def post_message(self, text: str, channel: str, thread_ts: str | None = None, **kwargs: object) -> bool:
        self.posts.append((text, {"channel": channel, "thread_ts": thread_ts, **kwargs}))
        return True


class MemoryLog:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def __call__(self, persona: str, role: str, text: str, ts: str, **kwargs: object) -> bool:
        self.rows.append((persona, role, text))
        return True


def _job() -> ScheduleJob:
    return ScheduleJob.from_dict(
        {
            "id": "job-a",
            "mode": "scheduled",
            "action": "fake",
            "notify": "slack_secretary",
            "every_day_at": "10:00",
            "persona": "persona-a",
            "memory": True,
        }
    )


def _run(slack: object, tmp_path: Path) -> MemoryLog:
    memory = MemoryLog()
    sch = PersonaScheduler(
        slack,  # type: ignore[arg-type]
        state_dir=tmp_path / "state",
        jobs=[_job()],
        repo_root=tmp_path,
        notify_channel_resolver=lambda job: "S1",
        append_memory_fn=memory,
        actions={"fake": lambda job: (True, "done text")},
    )
    sch.reload_jobs()
    sch._spawn_job(_job(), date(2026, 1, 5))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with sch._run_lock:
            if not sch._running:
                break
        time.sleep(0.01)
    return memory


def test_media_client_notify_and_memory(tmp_path: Path) -> None:
    media = FakeMedia()
    memory = _run(media, tmp_path)
    assert media.posts == [("done text", "S1", None)]
    assert [r[1] for r in memory.rows if r[1] == "assistant"] == ["assistant"]
    assert ("persona-a", "assistant", "done text") in memory.rows


def test_legacy_protocol_notify_and_memory(tmp_path: Path) -> None:
    legacy = LegacySlack()
    with pytest.warns(DeprecationWarning):
        memory = _run(legacy, tmp_path)
    assert legacy.posts == [("done text", {"channel": "S1", "thread_ts": None})]
    assert ("persona-a", "assistant", "done text") in memory.rows
