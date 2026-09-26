"""mltgnt.media._core.enqueue_guard (#4030)."""

from __future__ import annotations

from dataclasses import replace

from mltgnt.config.language import JA
from mltgnt.interfaces.media import Status
from mltgnt.media._core.enqueue_guard import enqueue_or_report


class FakeClient:
    def __init__(self, post_result: str | None = "p1") -> None:
        self.posts: list[tuple[str, str, str | None]] = []
        self.statuses: list[tuple[str, Status]] = []
        self.post_result = post_result

    def post(self, text: str, space: str, thread: str | None = None) -> str | None:
        self.posts.append((text, space, thread))
        return self.post_result

    def update(self, message_id: str, text: str) -> bool:
        return True

    def set_status(self, message_id: str, status: Status) -> bool:
        self.statuses.append((message_id, status))
        return True


def test_success_marks_working() -> None:
    client = FakeClient()
    calls: list[int] = []
    assert enqueue_or_report(lambda: calls.append(1), client=client, space="C", thread="T", message_id="m") is True
    assert calls == [1]
    assert client.statuses == [("m", Status.WORKING)]
    assert client.posts == []


def test_failure_replies_enqueue_failed_text_and_reports() -> None:
    client = FakeClient(post_result=None)
    failures: list[BaseException] = []
    pack = replace(JA, enqueue_failed_text="could not queue")

    def boom() -> None:
        raise RuntimeError("ref lock")

    ok = enqueue_or_report(
        boom, client=client, space="C", thread="T", message_id="m", language=pack, on_failure=failures.append
    )
    assert ok is False
    assert client.statuses == [("m", Status.FAILED)]
    assert client.posts == [("could not queue", "C", "T")]
    assert [str(e) for e in failures] == ["ref lock"]


def test_failing_on_failure_does_not_raise() -> None:
    client = FakeClient()

    def boom() -> None:
        raise RuntimeError("x")

    def bad_hook(exc: BaseException) -> None:
        raise ValueError("hook")

    assert enqueue_or_report(boom, client=client, space="C", thread=None, message_id="m", on_failure=bad_hook) is False
    assert client.posts == [(JA.enqueue_failed_text, "C", None)]
