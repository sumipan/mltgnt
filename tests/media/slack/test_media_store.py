"""mltgnt.media.slack.media_store (#4029)."""

from __future__ import annotations

import logging
from datetime import timedelta, timezone
from pathlib import Path

import pytest

from mltgnt.media.slack import media_store

# 2024-01-01T23:30:00Z: still Jan 1 in UTC, already Jan 2 at UTC+9.
_TS = "1704151800.000100"


class FakeFetch:
    def __init__(self, data: bytes = b"img", fail: set[str] | None = None) -> None:
        self.data = data
        self.fail = fail or set()
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str]) -> bytes:
        self.calls.append((url, headers))
        if url in self.fail:
            raise OSError("download failed")
        return self.data


def _file(file_id: str, **kwargs: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": file_id,
        "mimetype": "image/png",
        "size": 3,
        "name": "photo.png",
        "url_private_download": f"https://files.example.test/{file_id}",
    }
    base.update(kwargs)
    return base


def test_saves_under_dest_root_by_channel_and_local_date(tmp_path: Path) -> None:
    fetch = FakeFetch()
    event = {"channel": "C1", "ts": _TS, "files": [_file("F1")]}
    tz = timezone(timedelta(hours=9))
    saved = media_store.save_images(event, dest_root=tmp_path, tz=tz, token="xoxb-t", fetch=fetch)
    assert len(saved) == 1
    item = saved[0]
    assert item.path == tmp_path / "C1" / "2024-01-02" / "F1_photo.png"
    assert item.path.read_bytes() == b"img"
    assert (item.file_id, item.mimetype, item.original_name) == ("F1", "image/png", "photo.png")
    assert fetch.calls == [("https://files.example.test/F1", {"Authorization": "Bearer xoxb-t"})]


def test_tz_changes_date_directory(tmp_path: Path) -> None:
    event = {"channel": "C1", "ts": _TS, "files": [_file("F1")]}
    saved = media_store.save_images(event, dest_root=tmp_path, tz=timezone.utc, fetch=FakeFetch())
    assert saved[0].path.parent == tmp_path / "C1" / "2024-01-01"


def test_skips_and_failures(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    fetch = FakeFetch(fail={"https://files.example.test/F4"})
    event = {
        "channel": "C1",
        "ts": "bad",
        "files": [
            _file("F1", mimetype="text/plain"),
            _file("F2", size=100),
            _file("F3", url_private_download=None),
            _file("F4"),
            _file("F5", name=None, mimetype="image/jpeg", url_private="https://files.example.test/F5"),
        ],
    }
    with caplog.at_level(logging.INFO):
        saved = media_store.save_images(event, dest_root=tmp_path, tz=timezone.utc, fetch=fetch, max_bytes=10)
    assert [s.file_id for s in saved] == ["F5"]
    assert saved[0].path == tmp_path / "C1" / "1970-01-01" / "F5.jpg"
    assert fetch.calls[0][1] == {}


def test_no_files(tmp_path: Path) -> None:
    assert media_store.save_images({}, dest_root=tmp_path, tz=timezone.utc, fetch=FakeFetch()) == []


def test_write_failure_is_skipped(tmp_path: Path) -> None:
    blocker = tmp_path / "C1"
    blocker.write_text("not a dir", encoding="utf-8")
    event = {"channel": "C1", "ts": _TS, "files": [_file("F1")]}
    assert media_store.save_images(event, dest_root=tmp_path, tz=timezone.utc, fetch=FakeFetch()) == []


def test_safe_filename(tmp_path: Path) -> None:
    assert media_store.safe_filename("F1", "../../etc/passwd", ".png", tmp_path) == "F1_passwd.png"
    assert media_store.safe_filename("F1", "....", ".png", tmp_path) == "F1_F1.png"
    assert media_store.safe_filename("F1", 'a:b*c?"<>|.png', ".png", tmp_path) == "F1_abc.png"
    (tmp_path / "F1.png").write_bytes(b"")
    (tmp_path / "F1_1.png").write_bytes(b"")
    assert media_store.safe_filename("F1", None, ".png", tmp_path) == "F1_2.png"


def test_ext_from_mimetype() -> None:
    assert media_store.ext_from_mimetype("image/jpeg") == ".jpg"
    assert media_store.ext_from_mimetype("image/x-icon; q=1") == ".x-icon"
    assert media_store.ext_from_mimetype("image/") == ".bin"
