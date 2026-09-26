"""PersonaScheduler notifies through the Slack and WebChat clients (#4032)."""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pytest

from mltgnt.interfaces.media import MediaClient
from mltgnt.media.slack.client import SlackClient
from mltgnt.media.slack.config import SlackMediaConfig
from mltgnt.media.webchat.client import WebChatClient
from mltgnt.media.webchat.config import WebChatMediaConfig
from mltgnt.scheduler import PersonaScheduler
from mltgnt.scheduler.models import ScheduleJob
from tests.media.slack.fakes import FakeWebClient


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


def _run(client: MediaClient, space: str, tmp_path: Path) -> MemoryLog:
    memory = MemoryLog()
    sch = PersonaScheduler(
        client,
        state_dir=tmp_path / "state",
        jobs=[_job()],
        repo_root=tmp_path,
        notify_channel_resolver=lambda job: space,
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


@pytest.mark.parametrize("medium", ["slack", "webchat"])
def test_notify_reaches_the_medium_and_is_remembered(tmp_path: Path, medium: str) -> None:
    if medium == "slack":
        web = FakeWebClient()
        config = SlackMediaConfig(state_dir=tmp_path / "s", pending_dir=tmp_path / "p", events_dir=tmp_path / "e")
        client: MediaClient = SlackClient(web, config)
        memory = _run(client, "C1", tmp_path)
        assert [(c["channel"], c["text"]) for c in web.calls_of("chat_postMessage")] == [("C1", "done text")]
    else:
        wconfig = WebChatMediaConfig(
            state_dir=tmp_path / "s", pending_dir=tmp_path / "p", events_dir=tmp_path / "e", store_dir=tmp_path / "w"
        )
        webchat = WebChatClient(wconfig)
        memory = _run(webchat, wconfig.space_id, tmp_path)
        rows = webchat.store.read_all()
        assert [(r["author"], r["text"], r["thread_ts"]) for r in rows] == [("assistant", "done text", None)]
    assert ("persona-a", "user", "[schedule task: job-a]") in memory.rows
    assert ("persona-a", "assistant", "done text") in memory.rows
