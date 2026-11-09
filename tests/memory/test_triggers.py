"""Tests for mltgnt.memory.triggers (Issue #6238)."""
from __future__ import annotations

import dataclasses

from mltgnt.config.language import EN
from mltgnt.memory.triggers import forced_memory_tool, strip_trigger_words


def _pack(**kwargs: object):
    return dataclasses.replace(
        EN,
        remember_trigger_words=frozenset({"remember", "rememberplease"}),
        forget_trigger_words=frozenset({"forget", "forgetplease"}),
        **kwargs,
    )


def test_forced_memory_tool_prefers_forget() -> None:
    pack = _pack()
    assert forced_memory_tool("please remember but forget that", pack) == "forget"
    assert forced_memory_tool("plain question", pack) is None
    assert forced_memory_tool("", pack) is None


def test_strip_trigger_words_uses_longest_first() -> None:
    pack = _pack()
    assert strip_trigger_words("  rememberplease note  ", pack) == "note"
