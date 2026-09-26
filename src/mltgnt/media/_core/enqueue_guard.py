"""Guard a job enqueue: a failure is reported in the thread instead of being dropped."""

from __future__ import annotations

import logging
from collections.abc import Callable

from mltgnt.config.language import JA, LanguagePack
from mltgnt.interfaces.media import MediaClient, Status

__all__ = ["enqueue_or_report"]

_log = logging.getLogger(__name__)


def enqueue_or_report(
    enqueue: Callable[[], object],
    *,
    client: MediaClient,
    space: str,
    thread: str | None,
    message_id: str,
    language: LanguagePack = JA,
    on_failure: Callable[[BaseException], None] | None = None,
) -> bool:
    """Call ``enqueue``; mark ``message_id`` WORKING on success.

    On failure mark it FAILED, reply ``language.enqueue_failed_text`` in the
    thread and hand the exception to ``on_failure`` (audit, releasing the
    thread). Return True on success.
    """
    try:
        enqueue()
    except Exception as exc:  # noqa: BLE001 - last line of defence at the entrance
        _log.error("[enqueue_guard] enqueue failed space=%s message_id=%s: %s", space, message_id, exc, exc_info=True)
        client.set_status(message_id, Status.FAILED)
        if client.post(language.enqueue_failed_text, space, thread) is None:
            _log.warning("[enqueue_guard] failure notice post failed space=%s", space)
        if on_failure is not None:
            try:
                on_failure(exc)
            except Exception:
                _log.warning("[enqueue_guard] on_failure raised", exc_info=True)
        return False
    client.set_status(message_id, Status.WORKING)
    return True
