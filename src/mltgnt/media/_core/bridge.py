"""Media-agnostic flow of one turn: MediaEvent -> admit -> TurnHandler -> post -> finalize.

A reply is posted to the event's thread right away. A delegated task is saved as a
pending record (``state`` / ``space`` / ``thread`` / ``conversation_id`` /
``message_ids`` / ``persona_id``) and the conversation stays running until the
watchers call ``deliver_result``. Messages queued meanwhile run as the next turn.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from mltgnt.conversation import session_store, thread_queue
from mltgnt.interfaces.media import MediaClient, Status
from mltgnt.interfaces.turn import HistoryMessage, TurnHandler, TurnInput, TurnResult
from mltgnt.media._core import id_map, thread_reactions
from mltgnt.media._core.config import MediaConfig
from mltgnt.media._core.hooks import HookRegistry
from mltgnt.media._core.pending import PendingStore
from mltgnt.media._core.types import MediaEvent

__all__ = ["MediaBridge"]

_log = logging.getLogger(__name__)


def _thread_of(conversation_id: str) -> str | None:
    try:
        return id_map.resolve(conversation_id)[1]
    except ValueError:
        return None


class MediaBridge:
    """Run turns for ``handler`` on ``client``; ``deliver_result`` is the watchers' deliver target."""

    def __init__(
        self,
        client: MediaClient,
        handler: TurnHandler,
        config: MediaConfig,
        hooks: HookRegistry | None = None,
        *,
        pending_prefix: str = "pending-",
    ) -> None:
        self._client = client
        self._handler = handler
        self._hooks = hooks if hooks is not None else HookRegistry()
        self._pending = PendingStore.from_config(config, prefix=pending_prefix)

    def handle_event(self, event: MediaEvent) -> TurnResult | None:
        """Result of this event's turn; None when stopped by a hook, queued, rejected or failed."""
        if self._hooks.run_on_inbound(event):
            return None
        admission = thread_reactions.admit(
            event.conversation_id,
            event.text,
            client=self._client,
            message_id=event.message_id,
            author=event.author,
        )
        if not admission.proceed:
            return None
        turn = TurnInput(
            conversation_id=event.conversation_id,
            text=event.text,
            attachments=event.attachments,
            history=self._history(event.conversation_id),
        )
        result = self._run_turn(event.space_id, turn, (event.message_id,))
        if result is None or result.kind != "task" or not result.task_ref:
            self._finalize(event.space_id, event.conversation_id)
        return result

    def deliver_result(self, uid: str, body: str) -> str | None:
        """Post a delegated job's ``body`` to its thread and fire ``on_result``.

        Returns the posted message id; None when ``uid`` is unknown (already
        delivered) or the post failed (the pending record is kept for a retry).
        """
        pending = self._pending.consume(uid)
        if pending is None:
            return None
        space = str(pending.get("space") or "")
        thread = pending.get("thread") or None
        conversation_id = str(pending.get("conversation_id") or "")
        message_ids = [str(m) for m in pending.get("message_ids") or [] if m]
        posted = self._client.post(body, space, thread)
        if not posted:
            _log.warning("[bridge] result post failed uid=%s", uid)
            self._pending.save(uid, pending)
            return None
        self._set_status(message_ids, Status.DONE)
        session_store.append_turn(conversation_id, "assistant", body, pending.get("persona_id"))
        self._hooks.run_on_result(uid, body)
        self._finalize(space, conversation_id)
        return posted

    def _history(self, conversation_id: str) -> tuple[HistoryMessage, ...]:
        return tuple(
            HistoryMessage(role=str(t.get("role")), text=str(t.get("content") or ""), persona_id=t.get("persona"))
            for t in session_store.load_turns(conversation_id)
            if t.get("kind") == "turn"
        )

    def _set_status(self, message_ids: Sequence[str], status: Status) -> None:
        for message_id in message_ids:
            self._client.set_status(message_id, status)

    def _run_turn(self, space: str, turn: TurnInput, message_ids: Sequence[str]) -> TurnResult | None:
        """One handler call. Only a successfully delegated task leaves the conversation running."""
        session_store.append_turn(turn.conversation_id, "user", turn.text)
        turn = self._hooks.run_before_dispatch(turn)
        try:
            result = self._handler.handle(turn)
        except Exception:
            _log.exception("[bridge] handler failed conversation_id=%s", turn.conversation_id)
            self._set_status(message_ids, Status.FAILED)
            return None
        thread = _thread_of(turn.conversation_id)
        if result.kind == "task":
            if not result.task_ref:
                _log.warning("[bridge] task result without task_ref conversation_id=%s", turn.conversation_id)
                self._set_status(message_ids, Status.FAILED)
                return result
            self._pending.save(
                result.task_ref,
                {
                    "state": "running",
                    "space": space,
                    "thread": thread,
                    "conversation_id": turn.conversation_id,
                    "message_ids": list(message_ids),
                    "persona_id": turn.persona_id,
                },
            )
            self._set_status(message_ids, Status.WORKING)
            return result
        posted = self._client.post(result.text, space, thread)
        if posted:
            session_store.append_turn(turn.conversation_id, "assistant", result.text, turn.persona_id)
            self._set_status(message_ids, Status.DONE)
        else:
            _log.warning("[bridge] reply post failed conversation_id=%s", turn.conversation_id)
            self._set_status(message_ids, Status.FAILED)
        self._hooks.run_after_post(result, posted)
        return result

    def _finalize(self, space: str, conversation_id: str) -> None:
        """Release the conversation, running queued messages as further turns until idle or delegated."""
        while True:
            entries = thread_queue.finish_turn(thread_queue.storage_key(conversation_id))
            if entries is None:
                return
            message_ids = [str(e.get("ts")) for e in entries if e.get("ts")]
            thread_reactions.acknowledge_drained(self._client, message_ids)
            turn = TurnInput(
                conversation_id=conversation_id,
                text=thread_queue.build_composite_instruction(entries),
                history=self._history(conversation_id),
            )
            result = self._run_turn(space, turn, message_ids)
            if result is not None and result.kind == "task" and result.task_ref:
                return
