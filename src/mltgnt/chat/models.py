"""chat.models — interfaces.types からの re-export（後方互換）。"""
from __future__ import annotations

from mltgnt.interfaces.types import ChatInput, ChatOutput, Message

__all__ = ["ChatInput", "ChatOutput", "Message"]
