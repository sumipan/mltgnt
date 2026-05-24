"""mltgnt.scheduler.ghdag_bridge — 後方互換 re-export shim。

実装は mltgnt.bridges.ghdag_bridge に移動済み。
diary/scripts/preflight.py が importlib.import_module() でこのパスを参照するため残存する。
削除は diary 側 import 更新（follow-up issue）と同時に行う。
"""
from mltgnt.bridges.ghdag_bridge import (  # noqa: F401
    DagStep,
    _extract_result_filename,
    _order_to_result_filename,
    _topological_sort,
    enqueue_and_wait,
    enqueue_dag,
)
