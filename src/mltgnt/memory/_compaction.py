"""mltgnt.memory._compaction — 後方互換 re-export shim。

実装は mltgnt.memory.compaction にある。
"""
from __future__ import annotations

import warnings

import mltgnt.memory.compaction as _canonical

__all__ = list(_canonical.__all__)


def __getattr__(name: str):
    if hasattr(_canonical, name):
        warnings.warn(
            f"mltgnt.memory._compaction.{name} は非推奨です。"
            " mltgnt.memory.compaction から直接 import してください。"
            " v0.10 で削除予定。",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_canonical, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
