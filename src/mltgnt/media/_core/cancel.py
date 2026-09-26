"""Cancel a conversation's jobs from the thread.

Cancel words are ``LanguagePack.cancel_words``; stopping a job is the host's
``cancel_job`` (e.g. a DAG cancel), replying about the outcome is the caller's.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from enum import Enum
from pathlib import Path

from mltgnt.config.language import JA, LanguagePack
from mltgnt.interfaces.media import MediaClient, Status

__all__ = ["CancelOutcome", "find_pending_uids", "handle_cancel", "is_cancel_request"]

_log = logging.getLogger(__name__)


class CancelOutcome(str, Enum):
    CANCELLED_RUNNING = "cancelled_running"
    CANCELLED_QUEUED = "cancelled_queued"
    NOTHING_TO_CANCEL = "nothing_to_cancel"


def is_cancel_request(text: str, language: LanguagePack = JA) -> bool:
    """True when ``text`` equals or starts with a cancel word (case-insensitive)."""
    stripped = (text or "").strip().casefold()
    if not stripped:
        return False
    return any(stripped.startswith(word.casefold()) for word in language.cancel_words if word)


def find_pending_uids(pending_dir: Path, space: str, thread: str, *, prefix: str = "pending-") -> list[str]:
    """uids of pending records whose ``space`` / ``thread`` match."""
    if not pending_dir.is_dir():
        return []
    uids: list[str] = []
    for path in sorted(pending_dir.glob(f"{prefix}*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("space") == space and str(data.get("thread") or "") == thread:
            uid = path.stem.removeprefix(prefix)
            if uid:
                uids.append(uid)
    return uids


def handle_cancel(
    *,
    client: MediaClient,
    message_id: str,
    running_uids: Iterable[str],
    cancel_job: Callable[[str], bool],
    queued_cancel_ids: Iterable[str] = (),
    discard_queued: Callable[[str], None] | None = None,
) -> CancelOutcome:
    """Cancel running jobs, settle queued cancel messages and mark them CANCELLED.

    ``queued_cancel_ids`` are message ids of cancel requests still in the wait
    queue; each is marked CANCELLED and passed to ``discard_queued``.
    """
    cancelled_running = False
    for uid in running_uids:
        try:
            if cancel_job(uid):
                cancelled_running = True
        except Exception:
            _log.warning("[cancel] cancel_job failed uid=%s", uid, exc_info=True)
    queued = [mid for mid in queued_cancel_ids if mid]
    if not cancelled_running and not queued:
        return CancelOutcome.NOTHING_TO_CANCEL
    client.set_status(message_id, Status.CANCELLED)
    for mid in queued:
        client.set_status(mid, Status.CANCELLED)
        if discard_queued is not None:
            discard_queued(mid)
    return CancelOutcome.CANCELLED_RUNNING if cancelled_running else CancelOutcome.CANCELLED_QUEUED
