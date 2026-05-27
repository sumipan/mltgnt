"""mltgnt.chat.models — 後方互換 re-export shim。

正規の型定義は mltgnt.interfaces.types にある。
"""
from __future__ import annotations

import warnings

from mltgnt.interfaces.types import ChatInput as _ChatInput
from mltgnt.interfaces.types import ChatOutput as _ChatOutput
from mltgnt.interfaces.types import Message as _Message

_EXPORTS = {
    "ChatInput": _ChatInput,
    "ChatOutput": _ChatOutput,
    "Message": _Message,
}


def __getattr__(name: str):
    if name in _EXPORTS:
        warnings.warn(
            f"mltgnt.chat.models.{name} は非推奨です。"
            " mltgnt.interfaces.types から直接 import してください。"
            " v0.10 で削除予定。",
            DeprecationWarning,
            stacklevel=2,
        )
        return _EXPORTS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
