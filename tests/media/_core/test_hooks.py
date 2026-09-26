"""mltgnt.media._core.hooks (#4031)."""

from __future__ import annotations

import logging

import pytest

from mltgnt.interfaces.turn import TurnInput, TurnResult
from mltgnt.media._core.hooks import HookRegistry
from mltgnt.media._core.types import MediaEvent

EVENT = MediaEvent(space_id="C1", conversation_id="C1:1.0", message_id="1.0", author="U1", text="hi")
TURN = TurnInput(conversation_id="C1:1.0", text="hi")
RESULT = TurnResult(kind="reply", text="ok")


def _boom(*_args: object) -> None:
    raise RuntimeError("boom")


def test_empty_registry_is_a_no_op() -> None:
    hooks = HookRegistry()
    assert hooks.run_on_inbound(EVENT) is False
    assert hooks.run_before_dispatch(TURN) is TURN
    hooks.run_after_post(RESULT, "m1")
    hooks.run_on_result("u1", "body")


def test_registration_returns_the_function() -> None:
    hooks = HookRegistry()

    @hooks.on_inbound
    def handler(event: MediaEvent) -> bool:
        return False

    assert handler(EVENT) is False


def test_hooks_run_in_registration_order() -> None:
    hooks = HookRegistry()
    calls: list[str] = []
    hooks.after_post(lambda r, m: calls.append(f"a:{m}"))
    hooks.after_post(lambda r, m: calls.append(f"b:{m}"))
    hooks.on_result(lambda u, b: calls.append(f"c:{u}"))
    hooks.on_result(lambda u, b: calls.append(f"d:{b}"))
    hooks.run_after_post(RESULT, "m1")
    hooks.run_on_result("u1", "body")
    assert calls == ["a:m1", "b:m1", "c:u1", "d:body"]


def test_on_inbound_true_stops_later_hooks() -> None:
    hooks = HookRegistry()
    calls: list[str] = []
    hooks.on_inbound(lambda e: calls.append("first") is None and False)
    hooks.on_inbound(lambda e: calls.append("second") is None)
    hooks.on_inbound(lambda e: calls.append("third") is None)
    assert hooks.run_on_inbound(EVENT) is True
    assert calls == ["first", "second"]


def test_on_inbound_all_false_continues() -> None:
    hooks = HookRegistry()
    hooks.on_inbound(lambda e: False)
    hooks.on_inbound(lambda e: None)  # type: ignore[arg-type,return-value]
    assert hooks.run_on_inbound(EVENT) is False


def test_before_dispatch_chains_replacements() -> None:
    hooks = HookRegistry()
    hooks.before_dispatch(lambda t: TurnInput(conversation_id=t.conversation_id, text=t.text + "-a"))
    hooks.before_dispatch(lambda t: TurnInput(conversation_id=t.conversation_id, text=t.text + "-b"))
    assert hooks.run_before_dispatch(TURN).text == "hi-a-b"


def test_before_dispatch_keeps_turn_on_non_turn_return(caplog: pytest.LogCaptureFixture) -> None:
    hooks = HookRegistry()
    hooks.before_dispatch(lambda t: None)  # type: ignore[arg-type,return-value]
    with caplog.at_level(logging.WARNING):
        assert hooks.run_before_dispatch(TURN) is TURN
    assert "before_dispatch" in caplog.text


def test_exceptions_are_logged_and_later_hooks_still_run(caplog: pytest.LogCaptureFixture) -> None:
    hooks = HookRegistry()
    calls: list[str] = []
    hooks.on_inbound(_boom)  # type: ignore[arg-type]
    hooks.on_inbound(lambda e: calls.append("inbound") is None and False)
    hooks.before_dispatch(_boom)  # type: ignore[arg-type]
    hooks.before_dispatch(lambda t: TurnInput(conversation_id=t.conversation_id, text="replaced"))
    hooks.after_post(_boom)
    hooks.after_post(lambda r, m: calls.append("after_post"))
    hooks.on_result(_boom)
    hooks.on_result(lambda u, b: calls.append("on_result"))
    with caplog.at_level(logging.ERROR):
        assert hooks.run_on_inbound(EVENT) is False
        assert hooks.run_before_dispatch(TURN).text == "replaced"
        hooks.run_after_post(RESULT, None)
        hooks.run_on_result("u1", "body")
    assert calls == ["inbound", "after_post", "on_result"]
    assert caplog.text.count("RuntimeError: boom") == 4
