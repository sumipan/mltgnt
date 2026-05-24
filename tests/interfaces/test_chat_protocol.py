"""Tests for ChatPipelineProtocol structural subtyping (issue-1106)."""
from __future__ import annotations

from pathlib import Path

import pytest

from mltgnt.interfaces.chat import ChatPipelineProtocol
from mltgnt.interfaces.types import ChatInputBase, ChatOutputBase


class _ConformsChatPipeline:
    """ChatPipelineProtocol を満たす最小実装。"""

    def run(self, inp: ChatInputBase, repo_root: Path) -> ChatOutputBase: ...


class _MissingRun:
    """run メソッドを持たない — Protocol を満たさない。"""

    pass


def test_chat_pipeline_protocol_conforming() -> None:
    """run メソッドを持つオブジェクトは ChatPipelineProtocol を満たす。"""
    assert isinstance(_ConformsChatPipeline(), ChatPipelineProtocol)


def test_chat_pipeline_missing_run_fails() -> None:
    """run を持たないオブジェクトは ChatPipelineProtocol を満たさない。"""
    assert not isinstance(_MissingRun(), ChatPipelineProtocol)


def test_chat_pipeline_protocol_uses_base_types() -> None:
    """ChatPipelineProtocol.run のシグネチャが ChatInputBase / ChatOutputBase を参照する。"""
    import inspect

    hints = ChatPipelineProtocol.run.__annotations__
    assert hints.get("inp") is ChatInputBase or "ChatInputBase" in str(hints.get("inp"))
    assert hints.get("return") is ChatOutputBase or "ChatOutputBase" in str(hints.get("return"))
