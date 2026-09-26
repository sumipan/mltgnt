"""mltgnt.media.slack.app (#4029)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from mltgnt.media.slack import app as slack_app
from mltgnt.media.slack.config import SlackMediaConfig
from tests.media.slack.fakes import FakeBoltApp


def _config(tmp_path: Path) -> SlackMediaConfig:
    return SlackMediaConfig(
        state_dir=tmp_path / "s",
        pending_dir=tmp_path / "p",
        events_dir=tmp_path / "e",
        bot_token_env="TEST_BOT_TOKEN",
        app_token_env="TEST_APP_TOKEN",
    )


def test_build_app_uses_factory_and_token_from_env(tmp_path: Path) -> None:
    app = slack_app.build_app(_config(tmp_path), app_factory=FakeBoltApp, environ={"TEST_BOT_TOKEN": "xoxb-1"})
    assert isinstance(app, FakeBoltApp)
    assert app.kwargs == {"token": "xoxb-1"}


def test_build_app_reads_os_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_BOT_TOKEN", "xoxb-2")
    app = slack_app.build_app(_config(tmp_path), app_factory=FakeBoltApp)
    assert app.kwargs == {"token": "xoxb-2"}


def test_build_app_missing_token(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="TEST_BOT_TOKEN"):
        slack_app.build_app(_config(tmp_path), app_factory=FakeBoltApp, environ={})


def test_build_app_without_bolt_points_to_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "slack_bolt", None)
    with pytest.raises(ImportError, match=r"mltgnt\[slack\]"):
        slack_app.build_app(_config(tmp_path), environ={"TEST_BOT_TOKEN": "xoxb-1"})


def test_build_app_default_factory_is_bolt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    fake_module = types.ModuleType("slack_bolt")
    fake_module.App = FakeBoltApp  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slack_bolt", fake_module)
    app = slack_app.build_app(_config(tmp_path), environ={"TEST_BOT_TOKEN": "xoxb-1"})
    assert isinstance(app, FakeBoltApp)


def test_start_socket_mode(tmp_path: Path) -> None:
    started: list[tuple[Any, str]] = []

    class FakeHandler:
        def __init__(self, app: Any, app_token: str) -> None:
            self.app = app
            self.app_token = app_token

        def start(self) -> None:
            started.append((self.app, self.app_token))

    app = FakeBoltApp(token="xoxb-1")
    slack_app.start_socket_mode(
        app, _config(tmp_path), handler_factory=FakeHandler, environ={"TEST_APP_TOKEN": "xapp-1"}
    )
    assert started == [(app, "xapp-1")]


def test_start_socket_mode_without_bolt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "slack_bolt.adapter.socket_mode", None)
    with pytest.raises(ImportError, match=r"mltgnt\[slack\]"):
        slack_app.start_socket_mode(FakeBoltApp(), _config(tmp_path), environ={"TEST_APP_TOKEN": "xapp-1"})


def test_slack_package_exports_nothing() -> None:
    import mltgnt.media.slack

    assert mltgnt.media.slack.__all__ == []


def test_config_defaults(tmp_path: Path) -> None:
    from mltgnt.interfaces.media import Status

    config = SlackMediaConfig(state_dir=tmp_path, pending_dir=tmp_path, events_dir=tmp_path)
    assert config.chunk_max_chars == 3000
    assert config.bot_token_env and config.app_token_env
    assert set(config.status_reactions) == set(Status)
    with pytest.raises(TypeError):
        config.status_reactions[Status.DONE] = "x"  # type: ignore[index]
