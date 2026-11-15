"""mltgnt.media._core.thread_reactions (#4030)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mltgnt.conversation import thread_queue
from mltgnt.conversation.thread_queue import AdmitResult
from mltgnt.interfaces.media import Status
from mltgnt.media._core import thread_reactions


class FakeClient:
    def __init__(self) -> None:
        self.statuses: list[tuple[str, Status]] = []

    def post(self, text: str, space: str, thread: str | None = None) -> str | None:
        return None

    def update(self, message_id: str, text: str) -> bool:
        return True

    def set_status(self, message_id: str, status: Status) -> bool:
        self.statuses.append((message_id, status))
        return True


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("queued", Status.RECEIVED),
        ("running", Status.WORKING),
        ("accepted", Status.WORKING),
        ("rejected", Status.FAILED),
        ("other", None),
    ],
)
def test_status_for_admission(status: str, expected: Status | None) -> None:
    result = AdmitResult(proceed=False, queued=False, status=status)
    assert thread_reactions.status_for_admission(result) is expected


def test_admit_sets_status_from_real_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(thread_queue, "THREAD_QUEUE_DIR", tmp_path)
    client = FakeClient()
    first = thread_reactions.admit("C1:T1", "do it", client=client, message_id="1.0", author="u")
    second = thread_reactions.admit("C1:T1", "also this", client=client, message_id="2.0", author="u")
    assert (first.status, second.status) == ("accepted", "queued")
    assert client.statuses == [("1.0", Status.WORKING), ("2.0", Status.RECEIVED)]


def test_admit_skips_unknown_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_admit(*args: object, **kwargs: object) -> AdmitResult:
        return AdmitResult(proceed=False, queued=False, status="odd")

    monkeypatch.setattr(thread_queue, "admit", fake_admit)
    client = FakeClient()
    thread_reactions.admit("C:T", "x", client=client, message_id="1")
    assert client.statuses == []


def test_acknowledge_drained_marks_working() -> None:
    client = FakeClient()
    thread_reactions.acknowledge_drained(client, ["a", "", "b"])
    assert client.statuses == [("a", Status.WORKING), ("b", Status.WORKING)]
