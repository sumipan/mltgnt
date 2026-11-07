"""MatrixMediaConfig (#6101)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mltgnt.media.matrix.config import MatrixMediaConfig


def _base(**kwargs: object) -> MatrixMediaConfig:
    root = Path("/tmp/matrix-test")
    return MatrixMediaConfig(
        state_dir=root / "s",
        pending_dir=root / "p",
        events_dir=root / "e",
        homeserver_url="http://127.0.0.1:6167",
        user_id="@staff:x",
        **kwargs,
    )


def test_invalid_room_key_raises() -> None:
    with pytest.raises(ValueError, match="room key"):
        _base(rooms={"Staff": "!a:x"})


def test_http_example_com_rejected() -> None:
    root = Path("/tmp/matrix-test")
    with pytest.raises(ValueError, match="homeserver_url"):
        MatrixMediaConfig(
            state_dir=root / "s",
            pending_dir=root / "p",
            events_dir=root / "e",
            homeserver_url="http://example.com",
            user_id="@staff:x",
        )


def test_space_and_room_mapping() -> None:
    cfg = _base(rooms={"staff": "!a:x"})
    assert cfg.space_for_room("!a:x") == "matrix.staff"
    assert cfg.room_for_space("matrix.staff") == "!a:x"
