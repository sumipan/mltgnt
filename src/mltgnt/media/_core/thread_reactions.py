"""Reaction layer of the conversation wait queue: admit result -> message status."""

from __future__ import annotations

from collections.abc import Iterable

from mltgnt.conversation import thread_queue
from mltgnt.conversation.thread_queue import AdmitResult
from mltgnt.interfaces.media import MediaClient, Status

__all__ = ["acknowledge_drained", "admit", "status_for_admission"]

_ADMIT_STATUS = {
    "accepted": Status.WORKING,
    "running": Status.WORKING,
    "queued": Status.RECEIVED,
    "rejected": Status.FAILED,
}


def status_for_admission(result: AdmitResult) -> Status | None:
    return _ADMIT_STATUS.get(str(result.status))


def admit(
    conversation_id: str,
    instruction: str,
    *,
    client: MediaClient,
    message_id: str,
    author: str = "",
) -> AdmitResult:
    """``thread_queue.admit`` and show the outcome on ``message_id``."""
    result = thread_queue.admit(conversation_id, instruction, message_ts=message_id, author=author)
    status = status_for_admission(result)
    if status is not None:
        client.set_status(message_id, status)
    return result


def acknowledge_drained(client: MediaClient, message_ids: Iterable[str]) -> None:
    """Queued messages taken into the next turn are now being worked on."""
    for message_id in message_ids:
        if message_id:
            client.set_status(message_id, Status.WORKING)
