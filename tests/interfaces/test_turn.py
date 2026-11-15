"""Tests for mltgnt.interfaces.turn (issue #3286)."""
from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path
from typing import Any, get_args, get_type_hints

# Slack-specific terms that must not appear in field names (AC-6 / nexus layer boundaries)
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

# Source scan uses the same allow-set as nexus so role comment "user" is not a false positive
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


def test_turn_result_fields() -> None:
    from mltgnt.interfaces.turn import TurnResult

    fields = {f.name for f in dataclasses.fields(TurnResult)}
    assert fields == {"kind", "text", "task_ref", "reaction", "post_options"}

    hints = get_type_hints(TurnResult)
    assert set(get_args(hints["kind"])) == {"reply", "task"}
    assert hints["text"] is str
    assert hints["task_ref"] == str | None
    assert hints["reaction"] == str | None
    assert hints["post_options"] == dict[str, Any] | None


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
        assert not leaked, f"{cls.__name__} has Slack-specific fields: {sorted(leaked)}"


def test_turn_module_source_has_no_slack_field_names() -> None:
    src = _TURN_MODULE.read_text(encoding="utf-8")
    for name in _SLACK_SOURCE_NAMES:
        assert not re.search(rf"\b{re.escape(name)}\b", src), (
            f"turn.py contains Slack-specific term {name!r}"
        )


