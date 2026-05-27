"""mltgnt.scheduler.ghdag_bridge — 後方互換 re-export shim。

実装は mltgnt.bridges.ghdag_bridge に移動済み。
diary/scripts/preflight.py が importlib.import_module() でこのパスを参照するため残存する。
削除は diary 側 import 更新（follow-up issue）と同時に行う。
"""
from __future__ import annotations

import warnings

from mltgnt.bridges import ghdag_bridge as _canonical

_EXPORT_NAMES = (
    "DagStep",
    "_extract_result_filename",
    "_order_to_result_filename",
    "_topological_sort",
    "enqueue_and_wait",
    "enqueue_dag",
)


def __getattr__(name: str):
    if name in _EXPORT_NAMES:
        warnings.warn(
            f"mltgnt.scheduler.ghdag_bridge.{name} は非推奨です。"
            " mltgnt.bridges.ghdag_bridge から直接 import してください。"
            " v0.10 で削除予定。",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_canonical, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
