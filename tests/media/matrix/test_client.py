"""MatrixClient (#6101)."""

from __future__ import annotations

import ast
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from mltgnt.interfaces.media import MediaClient, Status
from mltgnt.media.matrix.client import MatrixClient, MatrixResponse
from mltgnt.media.matrix.config import MatrixMediaConfig
from tests.media.matrix.fakes import FakeHomeserver


def _config(tmp_path: Path) -> MatrixMediaConfig:
    return MatrixMediaConfig(
        state_dir=tmp_path / "s",
        pending_dir=tmp_path / "p",
        events_dir=tmp_path / "e",
        homeserver_url="http://127.0.0.1:8008",
        user_id="@staff:x",
        rooms={"staff": "!a:x"},
    )


def _client(tmp_path: Path, fake: FakeHomeserver) -> MatrixClient:
    return MatrixClient(_config(tmp_path), transport=fake)


def test_satisfies_media_client(tmp_path: Path) -> None:
    assert isinstance(_client(tmp_path, FakeHomeserver()), MediaClient)


def test_post_sends_text_to_room(tmp_path: Path) -> None:
    fake = FakeHomeserver()
    event_id = _client(tmp_path, fake).post("hi", "matrix.staff")
    assert event_id is not None
    sends = fake.calls_matching("/send/m.room.message/")
    assert len(sends) == 1
    assert sends[0].path.startswith("/_matrix/client/v3/rooms/%21a%3Ax/send/m.room.message/")
    body = sends[0].body
    assert isinstance(body, dict)
    assert body["msgtype"] == "m.text"
    assert body["body"] == "hi"
    assert body["format"] == "org.matrix.custom.html"


def test_post_unknown_space_returns_none(tmp_path: Path) -> None:
    assert _client(tmp_path, FakeHomeserver()).post("hi", "matrix.unknown") is None


def test_post_thread_relates_to(tmp_path: Path) -> None:
    fake = FakeHomeserver()
    _client(tmp_path, fake).post("hi", "matrix.staff", "$root")
    body = fake.send_bodies()[-1]
    rel = body["m.relates_to"]
    assert rel["rel_type"] == "m.thread"
    assert rel["event_id"] == "$root"
    assert rel["is_falling_back"] is True
    assert rel["m.in_reply_to"]["event_id"] == "$root"


def test_update_sends_replace(tmp_path: Path) -> None:
    fake = FakeHomeserver()
    client = _client(tmp_path, fake)
    event_id = client.post("hi", "matrix.staff")
    assert event_id is not None
    fake.calls.clear()
    assert client.update(event_id, "new") is True
    body = fake.send_bodies()[-1]
    assert body["m.relates_to"]["rel_type"] == "m.replace"
    assert body["m.new_content"]["body"] == "new"


def test_update_unknown_id_returns_false(tmp_path: Path) -> None:
    assert _client(tmp_path, FakeHomeserver()).update("$missing", "x") is False


def test_set_status_typing(tmp_path: Path) -> None:
    fake = FakeHomeserver()
    client = _client(tmp_path, fake)
    client.remember("$in", "!a:x")
    assert client.set_status("$in", Status.WORKING) is True
    path = fake.calls[-1].path
    assert "/typing/%40staff%3Ax" in path
    body = fake.calls[-1].body
    assert isinstance(body, dict) and body["typing"] is True and body["timeout"] == 30000
    fake.calls.clear()
    assert client.set_status("$in", Status.DONE) is True
    body2 = fake.calls[-1].body
    assert isinstance(body2, dict) and body2["typing"] is False


def test_react_emoji_map(tmp_path: Path) -> None:
    fake = FakeHomeserver()
    client = _client(tmp_path, fake)
    client.remember("$in", "!a:x")
    assert client.react("$in", "bulb") is True
    body = fake.calls[-1].body
    assert isinstance(body, dict)
    assert body["msgtype"] == "m.reaction"
    assert body["m.relates_to"]["key"] == "\U0001f4a1"
    assert body["m.relates_to"]["rel_type"] == "m.annotation"
    fake.calls.clear()
    assert client.react("$in", "no_such_emoji") is False
    assert fake.calls == []


def test_upload_png(tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 20)
    fake = FakeHomeserver()
    assert _client(tmp_path, fake).upload(str(img), "matrix.staff") is True
    assert any("/upload" in c.path for c in fake.calls)
    msg = fake.send_bodies()[-1]
    assert msg["msgtype"] == "m.image"
    assert msg["url"] == "mxc://x/y"


def test_upload_too_large(tmp_path: Path) -> None:
    cfg = MatrixMediaConfig(
        state_dir=tmp_path / "s",
        pending_dir=tmp_path / "p",
        events_dir=tmp_path / "e",
        homeserver_url="http://127.0.0.1:8008",
        user_id="@staff:x",
        rooms={"staff": "!a:x"},
        max_upload_bytes=5,
    )
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 10)
    assert MatrixClient(cfg, transport=FakeHomeserver()).upload(str(big), "matrix.staff") is False


def test_post_retries_once_on_429(tmp_path: Path) -> None:
    fake = FakeHomeserver()
    fake.queue("PUT", "/send/m.room.message/", MatrixResponse(429, {"retry_after_ms": 1500}))
    fake.queue("PUT", "/send/m.room.message/", MatrixResponse(200, {"event_id": "$ok"}))
    sleeper = Mock()
    client = MatrixClient(_config(tmp_path), transport=fake, sleep=sleeper)
    assert client.post("hi", "matrix.staff") == "$ok"
    sleeper.assert_called_once_with(1.5)
    assert len(fake.calls_matching("/send/m.room.message/")) == 2


def test_post_double_429_returns_none(tmp_path: Path) -> None:
    fake = FakeHomeserver()
    fake.queue("PUT", "/send/m.room.message/", MatrixResponse(429, {"retry_after_ms": 100}))
    fake.queue("PUT", "/send/m.room.message/", MatrixResponse(429, {"retry_after_ms": 100}))
    assert MatrixClient(_config(tmp_path), transport=fake, sleep=time.sleep).post("hi", "matrix.staff") is None


def test_matrix_modules_import_only_stdlib_and_mltgnt() -> None:
    root = Path(__file__).resolve().parents[3] / "src" / "mltgnt" / "media" / "matrix"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    assert name in ("mltgnt",) or name in __import__("sys").stdlib_module_names
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                assert node.module is not None
                root_name = node.module.split(".")[0]
                assert root_name in ("mltgnt",) or root_name in __import__("sys").stdlib_module_names
