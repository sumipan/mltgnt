"""Host hook points around one media turn.

Hooks of one kind run in registration order. An exception from a hook is logged
and ignored: later hooks and the turn itself keep going.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from mltgnt.interfaces.turn import TurnInput, TurnResult
from mltgnt.media._core.types import MediaEvent

__all__ = ["AfterPost", "BeforeDispatch", "HookRegistry", "OnInbound", "OnResult"]

_log = logging.getLogger(__name__)

# True stops the turn (the host handled the input itself)
OnInbound = Callable[[MediaEvent], bool]
BeforeDispatch = Callable[[TurnInput], TurnInput]
AfterPost = Callable[[TurnResult, "str | None"], None]
OnResult = Callable[[str, str], None]


class HookRegistry:
    """``on_inbound`` / ``before_dispatch`` / ``after_post`` / ``on_result`` hooks.

    Registration methods return the function, so they also work as decorators.
    """

    def __init__(self) -> None:
        self._on_inbound: list[OnInbound] = []
        self._before_dispatch: list[BeforeDispatch] = []
        self._after_post: list[AfterPost] = []
        self._on_result: list[OnResult] = []

    def on_inbound(self, fn: OnInbound) -> OnInbound:
        self._on_inbound.append(fn)
        return fn

    def before_dispatch(self, fn: BeforeDispatch) -> BeforeDispatch:
        self._before_dispatch.append(fn)
        return fn

    def after_post(self, fn: AfterPost) -> AfterPost:
        self._after_post.append(fn)
        return fn

    def on_result(self, fn: OnResult) -> OnResult:
        self._on_result.append(fn)
        return fn

    def run_on_inbound(self, event: MediaEvent) -> bool:
        """True as soon as one hook returns True (later hooks are skipped)."""
        for fn in list(self._on_inbound):
            try:
                if fn(event) is True:
                    return True
            except Exception:
                _log.exception("[hooks] on_inbound %r failed", fn)
        return False

    def run_before_dispatch(self, turn: TurnInput) -> TurnInput:
        """Pass ``turn`` through each hook; a failing or non-TurnInput hook leaves it unchanged."""
        for fn in list(self._before_dispatch):
            try:
                replaced = fn(turn)
            except Exception:
                _log.exception("[hooks] before_dispatch %r failed", fn)
                continue
            if isinstance(replaced, TurnInput):
                turn = replaced
            else:
                _log.warning("[hooks] before_dispatch %r returned %s; ignored", fn, type(replaced).__name__)
        return turn

    def run_after_post(self, result: TurnResult, message_id: str | None) -> None:
        for fn in list(self._after_post):
            try:
                fn(result, message_id)
            except Exception:
                _log.exception("[hooks] after_post %r failed", fn)

    def run_on_result(self, uid: str, body: str) -> None:
        for fn in list(self._on_result):
            try:
                fn(uid, body)
            except Exception:
                _log.exception("[hooks] on_result %r failed", fn)
