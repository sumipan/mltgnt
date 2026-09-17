"""Tests for interfaces/types.py — L1 DTO Protocol conformance (issue-1106)."""
from __future__ import annotations

from datetime import datetime


from mltgnt.interfaces.types import ChatInputBase, ChatOutputBase, Message, PersonaFMBase


# ---------------------------------------------------------------------------
# PersonaFMBase Protocol
# ---------------------------------------------------------------------------


class _ConformsPersonaFMBase:
    """Minimal implementation of PersonaFMBase."""

    name: str = "test-persona"


class _MissingName:
    """No name attribute — does not satisfy Protocol."""

    pass


def test_persona_fm_base_conforming() -> None:
    """Objects with name: str satisfy PersonaFMBase."""
    assert isinstance(_ConformsPersonaFMBase(), PersonaFMBase)


def test_persona_fm_base_missing_name_fails() -> None:
    """Objects without name do not satisfy PersonaFMBase."""
    assert not isinstance(_MissingName(), PersonaFMBase)


def test_real_persona_fm_conforms() -> None:
    """persona.schema.PersonaFM satisfies PersonaFMBase (structural subtyping)."""
    from mltgnt.persona.schema import PersonaFM

    fm = PersonaFM(name="real-persona")
    assert isinstance(fm, PersonaFMBase)


# ---------------------------------------------------------------------------
# Message TypedDict
# ---------------------------------------------------------------------------


def test_message_typed_dict_structure() -> None:
    """Message TypedDict has role / content keys."""
    msg: Message = {"role": "user", "content": "hello"}
    assert msg["role"] == "user"
    assert msg["content"] == "hello"


# ---------------------------------------------------------------------------
# ChatInputBase Protocol
# ---------------------------------------------------------------------------


class _ConformsChatInputBase:
    """Minimal implementation of ChatInputBase."""

    source: str = "slack"
    session_key: str = "sess-1"
    messages: list[Message] = []
    persona_name: str = "test"


class _MissingSessionKey:
    """Missing session_key — does not satisfy Protocol."""

    source: str = "slack"
    messages: list[Message] = []
    persona_name: str = "test"


def test_chat_input_base_conforming() -> None:
    """Objects with required attributes satisfy ChatInputBase."""
    assert isinstance(_ConformsChatInputBase(), ChatInputBase)


def test_chat_input_base_missing_session_key_fails() -> None:
    """Objects without session_key do not satisfy ChatInputBase."""
    assert not isinstance(_MissingSessionKey(), ChatInputBase)


def test_real_chat_input_conforms() -> None:
    """interfaces.types.ChatInput satisfies ChatInputBase (structural subtyping)."""
    from mltgnt.interfaces.types import ChatInput

    inp = ChatInput(source="slack", session_key="sess-1", messages=[], persona_name="test")
    assert isinstance(inp, ChatInputBase)


# ---------------------------------------------------------------------------
# ChatOutputBase Protocol
# ---------------------------------------------------------------------------


class _ConformsChatOutputBase:
    """Minimal implementation of ChatOutputBase."""

    content: str = "reply"
    persona_name: str = "test"
    timestamp: datetime = datetime(2024, 1, 1)
    session_key: str = "sess-1"


class _MissingTimestamp:
    """Missing timestamp — does not satisfy Protocol."""

    content: str = "reply"
    persona_name: str = "test"
    session_key: str = "sess-1"


def test_chat_output_base_conforming() -> None:
    """Objects with required attributes satisfy ChatOutputBase."""
    assert isinstance(_ConformsChatOutputBase(), ChatOutputBase)


def test_chat_output_base_missing_timestamp_fails() -> None:
    """Objects without timestamp do not satisfy ChatOutputBase."""
    assert not isinstance(_MissingTimestamp(), ChatOutputBase)


def test_real_chat_output_conforms() -> None:
    """interfaces.types.ChatOutput satisfies ChatOutputBase (structural subtyping)."""
    from mltgnt.interfaces.types import ChatOutput

    out = ChatOutput(
        content="reply",
        persona_name="test",
        timestamp=datetime(2024, 1, 1),
        session_key="sess-1",
    )
    assert isinstance(out, ChatOutputBase)
