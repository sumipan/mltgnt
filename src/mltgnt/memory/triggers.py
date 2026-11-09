"""mltgnt.memory.triggers — LanguagePack trigger word detection."""
from __future__ import annotations

from typing import Literal

from mltgnt.config.language import LanguagePack

__all__ = ["forced_memory_tool", "strip_trigger_words"]


def forced_memory_tool(text: str, pack: LanguagePack) -> Literal["remember", "forget"] | None:
    """Return a forced memory tool when trigger words appear (forget before remember)."""
    if not text:
        return None
    if any(word in text for word in pack.forget_trigger_words):
        return "forget"
    if any(word in text for word in pack.remember_trigger_words):
        return "remember"
    return None


def strip_trigger_words(text: str, pack: LanguagePack) -> str:
    """Remove trigger words (longest first) and strip surrounding whitespace."""
    out = text or ""
    words = sorted(
        pack.remember_trigger_words | pack.forget_trigger_words,
        key=len,
        reverse=True,
    )
    for word in words:
        out = out.replace(word, "")
    return out.strip()
