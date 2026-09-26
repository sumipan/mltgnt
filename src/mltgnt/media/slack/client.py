"""SlackClient: MediaClient implementation over a Slack WebClient.

Failures are logged and reported as None / False; no exception escapes.
"""

from __future__ import annotations

import logging
from typing import Any

from mltgnt.interfaces.media import Status
from mltgnt.media._core import id_map
from mltgnt.media.slack.config import SlackMediaConfig

__all__ = ["SlackClient", "split_text"]

_log = logging.getLogger(__name__)


def split_text(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars``, preferring newline boundaries."""
    chunks: list[str] = []
    rest = text
    while len(rest) > max_chars:
        cut = rest.rfind("\n", 0, max_chars + 1)
        if cut > 0:
            chunks.append(rest[:cut])
            rest = rest[cut + 1 :]
        else:
            chunks.append(rest[:max_chars])
            rest = rest[max_chars:]
    chunks.append(rest)
    return chunks


def _api_error(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    try:
        error = response.get("error") if response is not None else None
    except Exception:
        error = None
    return str(error or exc)


class SlackClient:
    """Post / update / react on Slack.

    ``message_id`` is either a message ``ts`` (its channel is taken from an earlier
    ``post`` or ``default_channel``) or a conversation id ``<channel>:<ts>``.
    """

    def __init__(self, web_client: Any, config: SlackMediaConfig, *, default_channel: str = "") -> None:
        self._web = web_client
        self._config = config
        self._default_channel = default_channel
        self._channels: dict[str, str] = {}

    def _locate(self, message_id: str) -> tuple[str, str] | None:
        if ":" in message_id:
            try:
                return id_map.resolve(message_id)
            except ValueError:
                return None
        channel = self._channels.get(message_id) or self._default_channel
        if not channel:
            _log.warning("slack client: unknown channel for message %s", message_id)
            return None
        return channel, message_id

    def post(self, text: str, space: str, thread: str | None = None) -> str | None:
        """Post ``text`` split by ``chunk_max_chars``; return the first chunk's ts.

        Later chunks go to ``thread``, or under the first chunk when ``thread`` is None.
        """
        first_ts: str | None = None
        for chunk in split_text(text, self._config.chunk_max_chars):
            kwargs: dict[str, Any] = {"channel": space, "text": chunk}
            reply_to = thread or first_ts
            if reply_to:
                kwargs["thread_ts"] = reply_to
            try:
                resp = self._web.chat_postMessage(**kwargs)
            except Exception as exc:
                _log.warning("slack client: chat_postMessage failed channel=%s: %s", space, _api_error(exc))
                return first_ts
            if first_ts is None:
                ts = resp.get("ts") if resp is not None else None
                if not isinstance(ts, str) or not ts.strip():
                    _log.warning("slack client: chat_postMessage returned no ts channel=%s", space)
                    return None
                first_ts = ts
                self._channels[ts] = space
        return first_ts

    def update(self, message_id: str, text: str) -> bool:
        located = self._locate(message_id)
        if located is None:
            return False
        channel, ts = located
        try:
            self._web.chat_update(channel=channel, ts=ts, text=text)
        except Exception as exc:
            _log.warning("slack client: chat_update failed channel=%s ts=%s: %s", channel, ts, _api_error(exc))
            return False
        return True

    def set_status(self, message_id: str, status: Status) -> bool:
        """Replace the other status reactions on ``message_id`` with the one for ``status``."""
        name = self._config.status_reactions.get(status)
        located = self._locate(message_id) if name else None
        if name is None or located is None:
            return False
        channel, ts = located
        for other in sorted(set(self._config.status_reactions.values()) - {name}):
            try:
                self._web.reactions_remove(channel=channel, timestamp=ts, name=other)
            except Exception as exc:
                _log.debug("slack client: reactions_remove %s ignored: %s", other, _api_error(exc))
        try:
            self._web.reactions_add(channel=channel, timestamp=ts, name=name)
        except Exception as exc:
            error = _api_error(exc)
            if error == "already_reacted":
                return True
            _log.warning("slack client: reactions_add %s failed channel=%s ts=%s: %s", name, channel, ts, error)
            return False
        return True

    def upload(self, path: str, space: str, thread: str | None = None) -> bool:
        """Unsupported: file posting is handled outside this client."""
        return False
