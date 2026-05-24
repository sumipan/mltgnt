"""Tests for interfaces/types.py — L1 DTO Protocol 適合テスト (issue-1106)."""
from __future__ import annotations

from datetime import datetime

import pytest

from mltgnt.interfaces.types import ChatInputBase, ChatOutputBase, Message, PersonaFMBase


# ---------------------------------------------------------------------------
# PersonaFMBase Protocol
# ---------------------------------------------------------------------------


class _ConformsPersonaFMBase:
    """PersonaFMBase を満たす最小実装。"""

    name: str = "test-persona"


class _MissingName:
    """name 属性を持たない — Protocol を満たさない。"""

    pass


def test_persona_fm_base_conforming() -> None:
    """name: str を持つオブジェクトは PersonaFMBase を満たす。"""
    assert isinstance(_ConformsPersonaFMBase(), PersonaFMBase)


def test_persona_fm_base_missing_name_fails() -> None:
    """name を持たないオブジェクトは PersonaFMBase を満たさない。"""
    assert not isinstance(_MissingName(), PersonaFMBase)


def test_real_persona_fm_conforms() -> None:
    """persona.schema.PersonaFM が PersonaFMBase を満たす（structural subtyping）。"""
    from mltgnt.persona.schema import PersonaFM

    fm = PersonaFM(name="real-persona")
    assert isinstance(fm, PersonaFMBase)


# ---------------------------------------------------------------------------
# Message TypedDict
# ---------------------------------------------------------------------------


def test_message_typed_dict_structure() -> None:
    """Message TypedDict が role / content キーを持つ。"""
    msg: Message = {"role": "user", "content": "hello"}
    assert msg["role"] == "user"
    assert msg["content"] == "hello"


# ---------------------------------------------------------------------------
# ChatInputBase Protocol
# ---------------------------------------------------------------------------


class _ConformsChatInputBase:
    """ChatInputBase を満たす最小実装。"""

    source: str = "slack"
    session_key: str = "sess-1"
    messages: list[Message] = []
    persona_name: str = "test"


class _MissingSessionKey:
    """session_key を欠く — Protocol を満たさない。"""

    source: str = "slack"
    messages: list[Message] = []
    persona_name: str = "test"


def test_chat_input_base_conforming() -> None:
    """必須属性を持つオブジェクトは ChatInputBase を満たす。"""
    assert isinstance(_ConformsChatInputBase(), ChatInputBase)


def test_chat_input_base_missing_session_key_fails() -> None:
    """session_key を持たないオブジェクトは ChatInputBase を満たさない。"""
    assert not isinstance(_MissingSessionKey(), ChatInputBase)


def test_real_chat_input_conforms() -> None:
    """chat.models.ChatInput が ChatInputBase を満たす（structural subtyping）。"""
    from mltgnt.chat.models import ChatInput

    inp = ChatInput(source="slack", session_key="sess-1", messages=[], persona_name="test")
    assert isinstance(inp, ChatInputBase)


# ---------------------------------------------------------------------------
# ChatOutputBase Protocol
# ---------------------------------------------------------------------------


class _ConformsChatOutputBase:
    """ChatOutputBase を満たす最小実装。"""

    content: str = "reply"
    persona_name: str = "test"
    timestamp: datetime = datetime(2024, 1, 1)
    session_key: str = "sess-1"


class _MissingTimestamp:
    """timestamp を欠く — Protocol を満たさない。"""

    content: str = "reply"
    persona_name: str = "test"
    session_key: str = "sess-1"


def test_chat_output_base_conforming() -> None:
    """必須属性を持つオブジェクトは ChatOutputBase を満たす。"""
    assert isinstance(_ConformsChatOutputBase(), ChatOutputBase)


def test_chat_output_base_missing_timestamp_fails() -> None:
    """timestamp を持たないオブジェクトは ChatOutputBase を満たさない。"""
    assert not isinstance(_MissingTimestamp(), ChatOutputBase)


def test_real_chat_output_conforms() -> None:
    """chat.models.ChatOutput が ChatOutputBase を満たす（structural subtyping）。"""
    from mltgnt.chat.models import ChatOutput

    out = ChatOutput(
        content="reply",
        persona_name="test",
        timestamp=datetime(2024, 1, 1),
        session_key="sess-1",
    )
    assert isinstance(out, ChatOutputBase)
