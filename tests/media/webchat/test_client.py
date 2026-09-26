"""mltgnt.media.webchat.client (#4032): the MediaClient contract over the store."""

from __future__ import annotations

from pathlib import Path

from mltgnt.interfaces.media import MediaClient, Status
from mltgnt.media.webchat.client import WebChatClient
from mltgnt.media.webchat.config import WebChatMediaConfig
from mltgnt.media.webchat.store import WebChatStore


def _config(tmp_path: Path) -> WebChatMediaConfig:
    return WebChatMediaConfig(
        state_dir=tmp_path / "s", pending_dir=tmp_path / "p", events_dir=tmp_path / "e", store_dir=tmp_path / "w"
    )


def test_config_defaults(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert (config.host, config.port, config.space_id) == ("127.0.0.1", 8765, "webchat")
    assert config.store_dir == tmp_path / "w"


def test_satisfies_media_client(tmp_path: Path) -> None:
    assert isinstance(WebChatClient(_config(tmp_path)), MediaClient)


def test_post_returns_hex_id_and_stores_the_message(tmp_path: Path) -> None:
    client = WebChatClient(_config(tmp_path))
    message_id = client.post("hello", "webchat", "t1")
    assert message_id is not None and len(message_id) == 32
    int(message_id, 16)
    row = client.store.latest(message_id)
    assert row is not None
    assert (row["author"], row["text"], row["thread_ts"], row["kind"]) == ("assistant", "hello", "t1", "message")


def test_post_without_thread_has_no_thread_ts(tmp_path: Path) -> None:
    client = WebChatClient(_config(tmp_path), author="bot")
    message_id = client.post("hello", "webchat")
    assert message_id is not None
    row = client.store.latest(message_id)
    assert row is not None and row["thread_ts"] is None and row["author"] == "bot"


def test_update_and_set_status_append_rows(tmp_path: Path) -> None:
    client = WebChatClient(_config(tmp_path))
    message_id = client.post("v1", "webchat")
    assert message_id is not None
    assert client.update(message_id, "v2") is True
    assert client.set_status(message_id, Status.DONE) is True
    rows = client.store.read(client.store.today())
    assert len(rows) == 1
    assert (rows[0]["text"], rows[0]["status"], rows[0]["kind"]) == ("v2", "done", "status")


def test_unknown_message_id_is_false(tmp_path: Path) -> None:
    client = WebChatClient(_config(tmp_path))
    assert client.update("missing", "x") is False
    assert client.set_status("missing", Status.WORKING) is False


def test_upload_is_unsupported(tmp_path: Path) -> None:
    assert WebChatClient(_config(tmp_path)).upload("file.txt", "webchat") is False


def test_io_failure_is_none_or_false(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("", encoding="utf-8")
    client = WebChatClient(_config(tmp_path), store=WebChatStore(blocker / "store"))
    assert client.post("hello", "webchat") is None


def test_revise_failure_is_false(tmp_path: Path) -> None:
    client = WebChatClient(_config(tmp_path))
    message_id = client.post("v1", "webchat")
    assert message_id is not None
    day_file = client.store.path_for(client.store.today())
    day_file.chmod(0o444)
    try:
        assert client.update(message_id, "v2") is False
    finally:
        day_file.chmod(0o644)
