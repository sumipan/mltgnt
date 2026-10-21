"""HubSubscriber polling and cursor persistence."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from mltgnt.daemon import DaemonComponent
from mltgnt.interfaces.hub import HubAddress, HubMessage
from mltgnt.interfaces.media import Status
from mltgnt.media._core.types import MediaEvent
from mltgnt.media.hub.config import HubMediaConfig
from mltgnt.media.hub.subscriber import HubSubscriber


def _config(tmp_path: Path, **kwargs: object) -> HubMediaConfig:
    defaults = {
        "state_dir": tmp_path / "s",
        "pending_dir": tmp_path / "p",
        "events_dir": tmp_path / "e",
        "hub_url": "http://127.0.0.1:8790",
        "host_id": "host-a",
        "token_env": "HUB_TOKEN",
        "rooms": ("r1",),
        "poll_interval_sec": 0.05,
    }
    defaults.update(kwargs)
    return HubMediaConfig(**defaults)  # type: ignore[arg-type]


def _msg(
    message_id: str,
    *,
    sender: HubAddress,
    to: HubAddress | None = None,
    kind: str = "message",
) -> HubMessage:
    return HubMessage(
        id=message_id,
        ts="2026-01-05T12:00:00+00:00",
        room="r1",
        thread=None,
        sender=sender,
        to=to,
        text="hello",
        attachments=(),
        hop=0,
        origin_id=None,
        kind=kind,  # type: ignore[arg-type]
        status=None,
    )


class ScriptedTransport:
    def __init__(self, batches: list[tuple[list[HubMessage], str | None] | None]) -> None:
        self._batches = list(batches)
        self.poll_calls = 0

    def send(self, message: HubMessage) -> str | None:
        return None

    def revise(
        self,
        room: str,
        message_id: str,
        *,
        kind: Literal["update", "status"],
        text: str | None = None,
        status: Status | None = None,
    ) -> bool:
        return False

    def poll(
        self,
        room: str,
        after: str | None,
        *,
        limit: int = 100,
    ) -> tuple[list[HubMessage], str | None] | None:
        self.poll_calls += 1
        if not self._batches:
            return [], None
        batch = self._batches.pop(0)
        return batch

    def guard_context(
        self,
        room: str,
        thread: str | None,
    ) -> tuple[list[Mapping[str, Any]], int] | None:
        return [], 0


class RecordingHandler:
    def __init__(self) -> None:
        self.events: list[MediaEvent] = []
        self.fail_ids: set[str] = set()
        self.lock = threading.Lock()

    def handle_event(self, event: MediaEvent) -> None:
        if event.message_id in self.fail_ids:
            raise RuntimeError("poison")
        with self.lock:
            self.events.append(event)


def _wait_for(predicate: Any, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def test_filters_self_other_host_and_non_message(tmp_path: Path) -> None:
    batches = [
        (
            [
                _msg("m1", sender=HubAddress("host-a", "bot")),
                _msg("m2", sender=HubAddress("host-b", None), to=HubAddress("host-c", None)),
                _msg("m3", sender=HubAddress("host-b", None), kind="status"),
                _msg("m4", sender=HubAddress("host-b", None)),
            ],
            "2026-01-05:4",
        ),
        ([], None),
    ]
    handler = RecordingHandler()
    transport = ScriptedTransport(batches)
    sub = HubSubscriber(_config(tmp_path), handler, transport=transport)
    sub.start()
    _wait_for(lambda: len(handler.events) == 1)
    sub.stop()
    assert handler.events[0].message_id == "m4"


def test_cursor_resume_and_no_duplicate(tmp_path: Path) -> None:
    msg = _msg("m1", sender=HubAddress("host-b", None))
    handler = RecordingHandler()
    transport = ScriptedTransport(
        [
            ([msg], "2026-01-05:1"),
            ([msg], "2026-01-05:1"),
            ([], None),
        ]
    )
    config = _config(tmp_path)
    sub = HubSubscriber(config, handler, transport=transport)
    sub.start()
    _wait_for(lambda: len(handler.events) == 1)
    sub.stop()
    sub2 = HubSubscriber(config, handler, transport=ScriptedTransport([([], None)]))
    sub2.start()
    time.sleep(0.15)
    sub2.stop()
    assert len(handler.events) == 1


def test_handle_event_exception_recorded_and_continues(tmp_path: Path) -> None:
    handler = RecordingHandler()
    handler.fail_ids.add("bad")
    batches = [
        (
            [
                _msg("bad", sender=HubAddress("host-b", None)),
                _msg("good", sender=HubAddress("host-b", None)),
            ],
            "2026-01-05:2",
        ),
        ([], None),
    ]
    sub = HubSubscriber(_config(tmp_path), handler, transport=ScriptedTransport(batches))
    sub.start()
    _wait_for(lambda: any(e.message_id == "good" for e in handler.events))
    sub.stop()
    state = json.loads((tmp_path / "s" / "hub-cursors" / "r1.json").read_text(encoding="utf-8"))
    assert "bad" in state["recent_ids"]


def test_corrupt_cursor_is_quarantined(tmp_path: Path) -> None:
    cursor_dir = tmp_path / "s" / "hub-cursors"
    cursor_dir.mkdir(parents=True)
    path = cursor_dir / "r1.json"
    path.write_text("not-json", encoding="utf-8")
    handler = RecordingHandler()
    transport = ScriptedTransport([([], None)])
    sub = HubSubscriber(_config(tmp_path), handler, transport=transport)
    sub.start()
    time.sleep(0.1)
    sub.stop()
    assert path.with_suffix(".json.corrupt").is_file()


def test_poll_failure_backoff_and_stop(tmp_path: Path) -> None:
    transport = ScriptedTransport([None, ([], None)])
    handler = RecordingHandler()
    sub = HubSubscriber(_config(tmp_path), handler, transport=transport)
    sub.start()
    _wait_for(lambda: transport.poll_calls >= 1)
    sub.stop()
    assert transport.poll_calls == 1


def test_daemon_component_contract(tmp_path: Path) -> None:
    sub = HubSubscriber(_config(tmp_path), RecordingHandler(), transport=ScriptedTransport([([], None)]))
    assert isinstance(sub, DaemonComponent)
    assert sub.name == "hub_subscriber"
    sub.start()
    sub.start()
    sub.stop()
    sub.stop()
