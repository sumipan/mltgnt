"""Slack Bolt App factory and Socket Mode start-up. Bolt is imported lazily."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from mltgnt.media.slack.config import SlackMediaConfig

__all__ = ["build_app", "start_socket_mode"]

_EXTRA_HINT = "slack_bolt is not installed; install the extra: pip install 'mltgnt[slack]'"


def _token(environ: Mapping[str, str] | None, name: str) -> str:
    env = os.environ if environ is None else environ
    token = env.get(name, "")
    if not token:
        raise RuntimeError(f"environment variable {name} is not set")
    return token


def build_app(
    config: SlackMediaConfig,
    *,
    app_factory: Callable[..., Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Create a Bolt ``App`` with the bot token read from ``config.bot_token_env``.

    ``app_factory`` replaces ``slack_bolt.App`` (tests pass a fake).
    """
    token = _token(environ, config.bot_token_env)
    if app_factory is None:
        try:
            from slack_bolt import App
        except ImportError as exc:
            raise ImportError(_EXTRA_HINT) from exc
        app_factory = App
    return app_factory(token=token)


def start_socket_mode(
    app: Any,
    config: SlackMediaConfig,
    *,
    handler_factory: Callable[[Any, str], Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Run ``app`` in Socket Mode with the token from ``config.app_token_env`` (blocking)."""
    app_token = _token(environ, config.app_token_env)
    if handler_factory is None:
        try:
            from slack_bolt.adapter.socket_mode import SocketModeHandler
        except ImportError as exc:
            raise ImportError(_EXTRA_HINT) from exc
        handler_factory = SocketModeHandler
    handler_factory(app, app_token).start()
