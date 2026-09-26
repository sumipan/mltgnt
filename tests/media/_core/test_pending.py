"""mltgnt.media._core.pending (#4029)."""

from __future__ import annotations

from pathlib import Path

from mltgnt.media._core.config import MediaConfig
from mltgnt.media._core.pending import PendingStore


def _config(tmp_path: Path) -> MediaConfig:
    return MediaConfig(state_dir=tmp_path / "s", pending_dir=tmp_path / "p", events_dir=tmp_path / "e")


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    store = PendingStore.from_config(_config(tmp_path))
    store.save("u1", {"channel": "C1", "text": "hello", "n": 2})
    assert store.load("u1") == {"channel": "C1", "text": "hello", "n": 2}
    assert (tmp_path / "p" / "pending-u1.json").is_file()


def test_load_missing_returns_none(tmp_path: Path) -> None:
    store = PendingStore(tmp_path / "missing")
    assert store.load("nope") is None


def test_load_corrupt_returns_none(tmp_path: Path) -> None:
    store = PendingStore(tmp_path, prefix="legacy-")
    (tmp_path / "legacy-u1.json").write_text("{broken", encoding="utf-8")
    assert store.load("u1") is None


def test_save_overwrites_and_leaves_no_temp_files(tmp_path: Path) -> None:
    store = PendingStore(tmp_path)
    store.save("u1", {"v": 1})
    store.save("u1", {"v": 2})
    assert store.load("u1") == {"v": 2}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pending-u1.json"]


def test_save_failure_removes_temp_file(tmp_path: Path) -> None:
    store = PendingStore(tmp_path)
    try:
        store.save("u1", {"v": object()})
    except TypeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected TypeError")
    assert list(tmp_path.iterdir()) == []


def test_delete(tmp_path: Path) -> None:
    store = PendingStore(tmp_path)
    store.save("u1", {"v": 1})
    assert store.delete("u1") is True
    assert store.delete("u1") is False
    assert store.load("u1") is None


def test_consume_returns_once(tmp_path: Path) -> None:
    store = PendingStore(tmp_path)
    store.save("u1", {"v": 1})
    assert store.consume("u1") == {"v": 1}
    assert store.consume("u1") is None
    assert list(tmp_path.iterdir()) == []


def test_consume_corrupt_returns_none(tmp_path: Path) -> None:
    store = PendingStore(tmp_path)
    (tmp_path / "pending-u1.json").write_text("{broken", encoding="utf-8")
    assert store.consume("u1") is None
    assert list(tmp_path.iterdir()) == []
