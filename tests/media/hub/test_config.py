"""HubMediaConfig validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mltgnt.media.hub.config import HubMediaConfig
from mltgnt.media.hub.guard import GuardConfig


def _base(tmp_path: Path, **kwargs: object) -> HubMediaConfig:
    defaults = {
        "state_dir": tmp_path / "s",
        "pending_dir": tmp_path / "p",
        "events_dir": tmp_path / "e",
        "hub_url": "http://127.0.0.1:8790",
        "host_id": "host-a",
        "token_env": "HUB_TOKEN",
        "rooms": ("r1",),
    }
    defaults.update(kwargs)
    return HubMediaConfig(**defaults)  # type: ignore[arg-type]


def test_defaults_and_resolved_cursor_dir(tmp_path: Path) -> None:
    config = _base(tmp_path)
    assert config.space_prefix == "hub."
    assert config.poll_interval_sec == 2.0
    assert config.guard == GuardConfig()
    assert config.resolved_cursor_dir == tmp_path / "s" / "hub-cursors"


def test_custom_cursor_dir(tmp_path: Path) -> None:
    custom = tmp_path / "cursors"
    config = _base(tmp_path, cursor_dir=custom)
    assert config.resolved_cursor_dir == custom


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hub_url": "ftp://x"},
        {"rooms": ()},
        {"rooms": ("../r1",)},
        {"poll_interval_sec": 0},
        {"space_prefix": ""},
        {"space_prefix": "hub:"},
        {"space_prefix": "hub/"},
    ],
)
def test_invalid_config_raises(tmp_path: Path, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _base(tmp_path, **kwargs)


def test_import_without_hub_extra() -> None:
    import importlib

    for mod in ("mltgnt.media.hub.config", "mltgnt.media.hub.client", "mltgnt.media.hub.inbound", "mltgnt.media.hub.subscriber"):
        importlib.import_module(mod)
