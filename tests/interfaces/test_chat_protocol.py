"""Tests for ChatPipelineProtocol structural subtyping (issue-1106)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from mltgnt.interfaces.types import ChatInput, ChatOutput, Message
from mltgnt.interfaces.chat import ChatPipelineProtocol
from mltgnt.interfaces.types import ChatInputBase, ChatOutputBase


class _ConformsChatPipeline:
    """Minimal implementation of ChatPipelineProtocol."""

    def run(self, inp: ChatInputBase, repo_root: Path) -> ChatOutputBase: ...


class _MissingRun:
    """No run method — does not satisfy Protocol."""

    pass


def test_chat_pipeline_protocol_conforming() -> None:
    """Objects with run satisfy ChatPipelineProtocol."""
    assert isinstance(_ConformsChatPipeline(), ChatPipelineProtocol)


def test_chat_pipeline_missing_run_fails() -> None:
    """Objects without run do not satisfy ChatPipelineProtocol."""
    assert not isinstance(_MissingRun(), ChatPipelineProtocol)


def test_chat_pipeline_protocol_uses_base_types() -> None:
    """ChatPipelineProtocol.run signature references ChatInputBase / ChatOutputBase."""

    hints = ChatPipelineProtocol.run.__annotations__
    assert hints.get("inp") is ChatInputBase or "ChatInputBase" in str(hints.get("inp"))
    assert hints.get("return") is ChatOutputBase or "ChatOutputBase" in str(hints.get("return"))


def test_chat_input_structurally_satisfies_chat_input_base() -> None:
    """ChatInput has all ChatInputBase attributes (structural subtyping)."""
    inp = ChatInput(
        source="slack",
        session_key="sess-1",
        messages=[Message(role="user", content="hello")],
        persona_name="test-persona",
    )
    assert isinstance(inp, ChatInputBase)


def test_chat_output_structurally_satisfies_chat_output_base() -> None:
    """ChatOutput has all ChatOutputBase attributes (structural subtyping)."""
    out = ChatOutput(
        content="reply",
        persona_name="test-persona",
        timestamp=datetime.now(timezone.utc),
        session_key="sess-1",
    )
    assert isinstance(out, ChatOutputBase)
