"""MediaClient / Status / adapt_client contract (#4027)."""
from __future__ import annotations

import warnings

import pytest

from mltgnt.interfaces.media import MediaClient, Status, adapt_client


class FakeMedia:
    """In-memory MediaClient."""

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
    """In-memory client exposing only the deprecated ``post_message``."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[tuple[tuple, dict]] = []

    def post_message(self, text: str, channel: str, thread_ts: str | None = None, **kwargs: object) -> bool:
        self.calls.append(((text,), {"channel": channel, "thread_ts": thread_ts, **kwargs}))
        return self.ok


class LegacySlackWithTs(LegacySlack):
    def post_message_ts(self, text: str, channel: str, thread_ts: str | None = None) -> str | None:
        self.calls.append(((text,), {"channel": channel, "thread_ts": thread_ts}))
        return "1700000000.000100" if self.ok else None


def test_status_values() -> None:
    assert [s.name for s in Status] == ["RECEIVED", "WORKING", "DONE", "FAILED", "CANCELLED"]
    assert all(isinstance(s, str) for s in Status)
    assert Status("done") is Status.DONE


def test_fake_media_satisfies_protocol() -> None:
    assert isinstance(FakeMedia(), MediaClient)


def test_upload_default_returns_false() -> None:
    class Explicit(MediaClient):
        def post(self, text: str, space: str, thread: str | None = None) -> str | None:
            return None

        def update(self, message_id: str, text: str) -> bool:
            return False

        def set_status(self, message_id: str, status: Status) -> bool:
            return False

    assert Explicit().upload("a.txt", "S1") is False


def test_adapt_client_returns_media_client_as_is_without_warning() -> None:
    client = FakeMedia()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert adapt_client(client) is client


def test_adapt_client_wraps_legacy_with_one_warning() -> None:
    legacy = LegacySlack()
    with pytest.warns(DeprecationWarning) as record:
        adapted = adapt_client(legacy)
    assert len([w for w in record if issubclass(w.category, DeprecationWarning)]) == 1

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = adapted.post("hi", "S1", "T1")

    assert legacy.calls == [(("hi",), {"channel": "S1", "thread_ts": "T1"})]
    assert result == ""


def test_adapter_returns_none_on_failure() -> None:
    with pytest.warns(DeprecationWarning):
        adapted = adapt_client(LegacySlack(ok=False))
    assert adapted.post("hi", "S1") is None


def test_adapter_prefers_post_message_ts() -> None:
    legacy = LegacySlackWithTs()
    with pytest.warns(DeprecationWarning):
        adapted = adapt_client(legacy)
    assert adapted.post("hi", "S1") == "1700000000.000100"
    assert len(legacy.calls) == 1


def test_adapter_other_methods_report_unsupported() -> None:
    with pytest.warns(DeprecationWarning):
        adapted = adapt_client(LegacySlack())
    assert adapted.update("M1", "x") is False
    assert adapted.set_status("M1", Status.DONE) is False
    assert adapted.upload("a.txt", "S1") is False


def test_adapt_client_rejects_unknown_object() -> None:
    with pytest.raises(TypeError):
        adapt_client(object())
