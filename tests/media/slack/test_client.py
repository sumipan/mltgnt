"""mltgnt.media.slack.client (#4029)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mltgnt.interfaces.media import MediaClient, Status
from mltgnt.media.slack.client import (
    POST_EXTRA_KEYS,
    UPDATE_EXTRA_KEYS,
    SlackClient,
    split_text,
)
from mltgnt.media.slack.config import SlackMediaConfig
from tests.media.slack.fakes import FakeSlackApiError, FakeWebClient


def _config(tmp_path: Path, **kwargs: object) -> SlackMediaConfig:
    return SlackMediaConfig(state_dir=tmp_path / "s", pending_dir=tmp_path / "p", events_dir=tmp_path / "e", **kwargs)


def _client(tmp_path: Path, web: FakeWebClient, **kwargs: object) -> SlackClient:
    return SlackClient(web, _config(tmp_path, **kwargs))


def test_satisfies_media_client(tmp_path: Path) -> None:
    assert isinstance(_client(tmp_path, FakeWebClient()), MediaClient)


def test_post_splits_long_text_and_returns_first_ts(tmp_path: Path) -> None:
    web = FakeWebClient()
    ts = _client(tmp_path, web).post("a" * 3001, "C1", "9.0")
    posts = web.calls_of("chat_postMessage")
    assert len(posts) == 2
    assert ts == "1.000100"
    assert [len(p["text"]) for p in posts] == [3000, 1]
    assert all(p["channel"] == "C1" and p["thread_ts"] == "9.0" for p in posts)


def test_post_short_text_without_thread(tmp_path: Path) -> None:
    web = FakeWebClient()
    assert _client(tmp_path, web).post("hi", "C1") == "1.000100"
    assert web.calls_of("chat_postMessage") == [{"channel": "C1", "text": "hi"}]


def test_post_later_chunks_thread_under_first(tmp_path: Path) -> None:
    web = FakeWebClient()
    _client(tmp_path, web, chunk_max_chars=5).post("abcdefgh", "C1")
    posts = web.calls_of("chat_postMessage")
    assert posts[0] == {"channel": "C1", "text": "abcde"}
    assert posts[1] == {"channel": "C1", "text": "fgh", "thread_ts": "1.000100"}


def test_post_failure_returns_none(tmp_path: Path) -> None:
    web = FakeWebClient(errors={"chat_postMessage": FakeSlackApiError("channel_not_found")})
    assert _client(tmp_path, web).post("hi", "C1") is None


def test_post_missing_ts_returns_none(tmp_path: Path) -> None:
    class NoTs(FakeWebClient):
        def chat_postMessage(self, **kwargs: object) -> dict[str, object]:  # noqa: N802
            super().chat_postMessage(**kwargs)
            return {"ok": True, "ts": ""}

    assert _client(tmp_path, NoTs()).post("hi", "C1") is None


def test_post_later_chunk_failure_keeps_first_ts(tmp_path: Path) -> None:
    class FailSecond(FakeWebClient):
        def chat_postMessage(self, **kwargs: object) -> dict[str, object]:  # noqa: N802
            if self.calls:
                raise FakeSlackApiError("rate_limited")
            return super().chat_postMessage(**kwargs)

    assert _client(tmp_path, FailSecond(), chunk_max_chars=2).post("abcd", "C1") == "1.000100"


def test_update(tmp_path: Path) -> None:
    web = FakeWebClient()
    client = _client(tmp_path, web)
    ts = client.post("hi", "C1")
    assert ts is not None
    assert client.update(ts, "edited") is True
    assert client.update("C2:5.5", "other") is True
    assert web.calls_of("chat_update") == [
        {"channel": "C1", "ts": ts, "text": "edited"},
        {"channel": "C2", "ts": "5.5", "text": "other"},
    ]


def test_update_unknown_channel_or_failure_returns_false(tmp_path: Path) -> None:
    web = FakeWebClient(errors={"chat_update": FakeSlackApiError("message_not_found")})
    client = _client(tmp_path, web)
    assert client.update("7.7", "x") is False
    assert web.calls == []
    assert client.update("C1:7.7", "x") is False


def test_default_channel_is_used_for_bare_ts(tmp_path: Path) -> None:
    web = FakeWebClient()
    client = SlackClient(web, _config(tmp_path), default_channel="C9")
    assert client.update("7.7", "x") is True
    assert web.calls_of("chat_update") == [{"channel": "C9", "ts": "7.7", "text": "x"}]


def test_set_status_swaps_reactions(tmp_path: Path) -> None:
    web = FakeWebClient()
    config = _config(tmp_path)
    client = SlackClient(web, config)
    ts = client.post("hi", "C1")
    assert ts is not None
    assert client.set_status(ts, Status.DONE) is True
    removed = [c["name"] for c in web.calls_of("reactions_remove")]
    added = web.calls_of("reactions_add")
    assert config.status_reactions[Status.WORKING] in removed
    assert config.status_reactions[Status.DONE] not in removed
    assert added == [{"channel": "C1", "timestamp": ts, "name": config.status_reactions[Status.DONE]}]


def test_set_status_on_conversation_id(tmp_path: Path) -> None:
    web = FakeWebClient()
    assert _client(tmp_path, web).set_status("C1:3.3", Status.WORKING) is True
    assert web.calls_of("reactions_add")[0]["channel"] == "C1"


def test_set_status_failures_return_false(tmp_path: Path) -> None:
    web = FakeWebClient(
        errors={
            "reactions_add": FakeSlackApiError("invalid_name"),
            "reactions_remove": FakeSlackApiError("no_reaction"),
        }
    )
    client = _client(tmp_path, web)
    assert client.set_status("C1:3.3", Status.DONE) is False
    assert client.set_status("3.3", Status.DONE) is False


def test_set_status_already_reacted_is_success(tmp_path: Path) -> None:
    web = FakeWebClient(errors={"reactions_add": FakeSlackApiError("already_reacted")})
    assert _client(tmp_path, web).set_status("C1:3.3", Status.DONE) is True


def test_set_status_unmapped_returns_false(tmp_path: Path) -> None:
    web = FakeWebClient()
    client = _client(tmp_path, web, status_reactions={Status.DONE: "white_check_mark"})
    assert client.set_status("C1:3.3", Status.FAILED) is False
    assert web.calls == []


def test_post_with_extra_passes_kwargs(tmp_path: Path) -> None:
    web = FakeWebClient()
    blocks = [{"type": "section", "text": {"type": "plain_text", "text": "hi"}}]
    ts = _client(tmp_path, web).post(
        "hi",
        "C1",
        extra={"username": "bot", "icon_emoji": ":robot:", "blocks": blocks},
    )
    assert ts == "1.000100"
    assert web.calls_of("chat_postMessage") == [
        {
            "channel": "C1",
            "text": "hi",
            "username": "bot",
            "icon_emoji": ":robot:",
            "blocks": blocks,
        }
    ]


def test_post_extra_on_later_chunks_only_username(tmp_path: Path) -> None:
    web = FakeWebClient()
    blocks = [{"type": "section"}]
    _client(tmp_path, web, chunk_max_chars=5).post(
        "abcdefgh",
        "C1",
        extra={"username": "bot", "blocks": blocks, "reply_broadcast": True},
    )
    posts = web.calls_of("chat_postMessage")
    assert posts[0] == {
        "channel": "C1",
        "text": "abcde",
        "username": "bot",
        "blocks": blocks,
        "reply_broadcast": True,
    }
    assert posts[1] == {"channel": "C1", "text": "fgh", "thread_ts": "1.000100", "username": "bot"}


def test_post_unknown_extra_raises(tmp_path: Path) -> None:
    web = FakeWebClient()
    with pytest.raises(ValueError, match="unknown"):
        _client(tmp_path, web).post("hi", "C1", extra={"unknown": 1})
    assert web.calls == []


def test_post_without_extra_unchanged(tmp_path: Path) -> None:
    web = FakeWebClient()
    assert _client(tmp_path, web).post("hi", "C1") == "1.000100"
    assert web.calls_of("chat_postMessage") == [{"channel": "C1", "text": "hi"}]


def test_update_with_extra_blocks(tmp_path: Path) -> None:
    web = FakeWebClient()
    blocks = [{"type": "section"}]
    client = _client(tmp_path, web)
    assert client.update("C1:1.2", "new", extra={"blocks": blocks}) is True
    assert web.calls_of("chat_update") == [
        {"channel": "C1", "ts": "1.2", "text": "new", "blocks": blocks},
    ]


def test_update_without_extra_unchanged(tmp_path: Path) -> None:
    web = FakeWebClient()
    client = _client(tmp_path, web)
    ts = client.post("hi", "C1")
    assert ts is not None
    assert client.update(ts, "edited") is True
    assert web.calls_of("chat_update") == [{"channel": "C1", "ts": ts, "text": "edited"}]


def test_update_unknown_extra_raises(tmp_path: Path) -> None:
    web = FakeWebClient()
    with pytest.raises(ValueError, match="username"):
        _client(tmp_path, web).update("C1:1.2", "new", extra={"username": "x"})
    assert web.calls == []


def test_upload_with_thread_and_title(tmp_path: Path) -> None:
    web = FakeWebClient()
    assert _client(tmp_path, web).upload("/tmp/a.png", "C1", thread="1.2", title="t") is True
    assert web.calls_of("files_upload_v2") == [
        {
            "channel": "C1",
            "file": "/tmp/a.png",
            "filename": "a.png",
            "thread_ts": "1.2",
            "title": "t",
        }
    ]


def test_upload_without_thread(tmp_path: Path) -> None:
    web = FakeWebClient()
    assert _client(tmp_path, web).upload("/tmp/a.png", "C1") is True
    assert web.calls_of("files_upload_v2") == [
        {"channel": "C1", "file": "/tmp/a.png", "filename": "a.png", "title": "a.png"}
    ]


def test_upload_failure_returns_false(tmp_path: Path) -> None:
    web = FakeWebClient(errors={"files_upload_v2": FakeSlackApiError("file_upload_failed")})
    assert _client(tmp_path, web).upload("/tmp/a.png", "C1") is False


def test_unreact_success(tmp_path: Path) -> None:
    web = FakeWebClient()
    assert _client(tmp_path, web).unreact("C1:1.2", "eyes") is True
    assert web.calls_of("reactions_remove") == [
        {"channel": "C1", "timestamp": "1.2", "name": "eyes"},
    ]


def test_unreact_no_reaction_is_success(tmp_path: Path) -> None:
    web = FakeWebClient(errors={"reactions_remove": FakeSlackApiError("no_reaction")})
    assert _client(tmp_path, web).unreact("C1:1.2", "eyes") is True


def test_unreact_other_error_returns_false(tmp_path: Path) -> None:
    web = FakeWebClient(errors={"reactions_remove": FakeSlackApiError("invalid_name")})
    assert _client(tmp_path, web).unreact("C1:1.2", "eyes") is False


def test_unreact_unknown_channel_returns_false(tmp_path: Path) -> None:
    web = FakeWebClient()
    assert _client(tmp_path, web).unreact("7.7", "eyes") is False
    assert web.calls == []


def test_extra_keys_constants() -> None:
    assert frozenset(
        {"username", "icon_emoji", "icon_url", "blocks", "reply_broadcast"}
    ) == POST_EXTRA_KEYS
    assert frozenset({"blocks"}) == UPDATE_EXTRA_KEYS


def test_split_text_prefers_newlines() -> None:
    assert split_text("", 5) == [""]
    assert split_text("abc", 5) == ["abc"]
    assert split_text("ab\ncdefg", 5) == ["ab", "cdefg"]
    assert split_text("abcdefg", 3) == ["abc", "def", "g"]
    assert "".join(split_text("x\n" * 40, 7)).count("x") == 40
