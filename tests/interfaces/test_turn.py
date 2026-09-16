"""Tests for mltgnt.interfaces.turn (issue #3286)."""
from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path
from typing import get_args, get_type_hints

import pytest

# フィールド名に含めてはいけない Slack 固有語（AC-6 / nexus test_layer_boundaries 相当）
_SLACK_FIELD_NAMES = frozenset(
    {
        "channel",
        "thread_ts",
        "user",
        "user_id",
        "blocks",
        "channel_id",
        "slack_channel_id",
        "slack_thread_ts",
    }
)

# ソース検査は role コメントの "user" を誤検知しないよう、nexus と同集合
_SLACK_SOURCE_NAMES = frozenset(
    {
        "channel",
        "thread_ts",
        "user_id",
        "blocks",
        "channel_id",
        "slack_channel_id",
        "slack_thread_ts",
    }
)

_TURN_MODULE = (
    Path(__file__).resolve().parents[2] / "src" / "mltgnt" / "interfaces" / "turn.py"
)


def test_package_exports_turn_types() -> None:
    from mltgnt.interfaces import TurnHandler, TurnInput, TurnResult
    from mltgnt.interfaces.turn import Attachment, HistoryMessage

    assert TurnInput is not None
    assert TurnResult is not None
    assert TurnHandler is not None
    assert Attachment is not None
    assert HistoryMessage is not None


def test_turn_input_fields_match_nexus_conversation_types() -> None:
    from mltgnt.interfaces.turn import Attachment, HistoryMessage, TurnInput

    fields = {f.name: f for f in dataclasses.fields(TurnInput)}
    assert set(fields) == {
        "conversation_id",
        "text",
        "attachments",
        "history",
        "persona_id",
    }
    assert fields["conversation_id"].type is str or fields["conversation_id"].type == "str"
    assert fields["text"].type is str or fields["text"].type == "str"

    hints = get_type_hints(TurnInput)
    assert hints["conversation_id"] is str
    assert hints["text"] is str
    assert hints["attachments"] == tuple[Attachment, ...]
    assert hints["history"] == tuple[HistoryMessage, ...]
    assert hints["persona_id"] == str | None


def test_turn_result_fields() -> None:
    from mltgnt.interfaces.turn import TurnResult

    fields = {f.name for f in dataclasses.fields(TurnResult)}
    assert fields == {"kind", "text", "task_ref"}

    hints = get_type_hints(TurnResult)
    assert set(get_args(hints["kind"])) == {"reply", "task"}
    assert hints["text"] is str
    assert hints["task_ref"] == str | None


def test_all_boundary_types_are_frozen_dataclasses() -> None:
    from mltgnt.interfaces.turn import Attachment, HistoryMessage, TurnInput, TurnResult

    for cls in (Attachment, HistoryMessage, TurnInput, TurnResult):
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} must be a dataclass"
        assert cls.__dataclass_params__.frozen, f"{cls.__name__} must be frozen"


def test_turn_handler_is_runtime_checkable_protocol() -> None:
    from typing import Protocol

    from mltgnt.interfaces.turn import TurnHandler, TurnInput, TurnResult

    assert issubclass(TurnHandler, Protocol)
    assert getattr(TurnHandler, "_is_runtime_protocol", False) is True

    params = list(inspect.signature(TurnHandler.handle).parameters)
    assert "turn" in params
    hints = get_type_hints(TurnHandler.handle)
    assert hints.get("turn") is TurnInput
    assert hints.get("return") is TurnResult

    class _Conforms:
        def handle(self, turn: TurnInput) -> TurnResult:
            return TurnResult(kind="reply", text=turn.text)

    class _Missing:
        pass

    assert isinstance(_Conforms(), TurnHandler)
    assert not isinstance(_Missing(), TurnHandler)


def test_no_slack_specific_field_names_on_types() -> None:
    from mltgnt.interfaces.turn import Attachment, HistoryMessage, TurnInput, TurnResult

    for cls in (Attachment, HistoryMessage, TurnInput, TurnResult):
        names = {f.name for f in dataclasses.fields(cls)}
        leaked = names & _SLACK_FIELD_NAMES
        assert not leaked, f"{cls.__name__} に Slack 固有フィールド: {sorted(leaked)}"


def test_turn_module_source_has_no_slack_field_names() -> None:
    src = _TURN_MODULE.read_text(encoding="utf-8")
    for name in _SLACK_SOURCE_NAMES:
        assert not re.search(rf"\b{re.escape(name)}\b", src), (
            f"turn.py に Slack 固有語 {name!r} がある"
        )


def test_attachment_and_history_message_defaults() -> None:
    from mltgnt.interfaces.turn import Attachment, HistoryMessage, TurnInput, TurnResult

    att = Attachment(name="a.txt")
    assert att.content_type is None
    assert att.uri is None

    msg = HistoryMessage(role="user", text="hi")
    assert msg.persona_id is None

    turn = TurnInput(conversation_id="c1", text="hello")
    assert turn.attachments == ()
    assert turn.history == ()
    assert turn.persona_id is None

    result = TurnResult(kind="task", task_ref="uuid-1")
    assert result.text == ""
    assert result.task_ref == "uuid-1"

    with pytest.raises(dataclasses.FrozenInstanceError):
        turn.text = "mutated"  # type: ignore[misc]
