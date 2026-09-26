"""mltgnt.media._core.cancel (#4030)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from mltgnt.config.language import JA
from mltgnt.interfaces.media import Status
from mltgnt.media._core.cancel import CancelOutcome, find_pending_uids, handle_cancel, is_cancel_request


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


def test_is_cancel_request_uses_language_cancel_words() -> None:
    assert is_cancel_request("cancel")
    assert is_cancel_request("  STOP now")
    assert not is_cancel_request("please continue")
    assert not is_cancel_request("")
    pack = replace(JA, cancel_words=frozenset({"abort"}))
    assert is_cancel_request("Abort!", pack)
    assert not is_cancel_request("cancel", pack)


def test_find_pending_uids(tmp_path: Path) -> None:
    assert find_pending_uids(tmp_path / "missing", "C", "T") == []
    (tmp_path / "pending-a.json").write_text(json.dumps({"space": "C", "thread": "T"}), encoding="utf-8")
    (tmp_path / "pending-b.json").write_text(json.dumps({"space": "C", "thread": "X"}), encoding="utf-8")
    (tmp_path / "pending-c.json").write_text("{bad", encoding="utf-8")
    (tmp_path / "pending-d.json").write_text("[]", encoding="utf-8")
    assert find_pending_uids(tmp_path, "C", "T") == ["a"]


def test_cancel_running_job() -> None:
    client = FakeClient()
    cancelled: list[str] = []

    def cancel_job(uid: str) -> bool:
        cancelled.append(uid)
        if uid == "bad":
            raise OSError("x")
        return uid == "u2"

    outcome = handle_cancel(client=client, message_id="m", running_uids=["bad", "u1", "u2"], cancel_job=cancel_job)
    assert outcome is CancelOutcome.CANCELLED_RUNNING
    assert cancelled == ["bad", "u1", "u2"]
    assert client.statuses == [("m", Status.CANCELLED)]


def test_cancel_queued_only() -> None:
    client = FakeClient()
    discarded: list[str] = []
    outcome = handle_cancel(
        client=client,
        message_id="m",
        running_uids=[],
        cancel_job=lambda uid: True,
        queued_cancel_ids=["q1", ""],
        discard_queued=discarded.append,
    )
    assert outcome is CancelOutcome.CANCELLED_QUEUED
    assert client.statuses == [("m", Status.CANCELLED), ("q1", Status.CANCELLED)]
    assert discarded == ["q1"]


def test_nothing_to_cancel() -> None:
    client = FakeClient()
    outcome = handle_cancel(client=client, message_id="m", running_uids=["u"], cancel_job=lambda uid: False)
    assert outcome is CancelOutcome.NOTHING_TO_CANCEL
    assert client.statuses == []
