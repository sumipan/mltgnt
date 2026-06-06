"""KPI 算出ロジック。"""
from __future__ import annotations

from collections import defaultdict

_TASK_COMPLETE = frozenset({"task_complete"})
_TASK_FAILED = frozenset({"task_failed"})
_TASK_EXIT = _TASK_COMPLETE | _TASK_FAILED
_TASK_COMPLETION = _TASK_EXIT
_MEMORY_HIT = frozenset({"memory_hit"})
_MEMORY_MISS = frozenset({"memory_miss"})


def response_failure_rate(records: list[dict]) -> tuple[float, tuple[int, int]]:
    """応答失敗率と (failed, total) を返す。total==0 のとき rate は 0.0。"""
    failed = sum(1 for r in records if r.get("event_type") in _TASK_FAILED)
    complete = sum(1 for r in records if r.get("event_type") in _TASK_COMPLETE)
    total = complete + failed
    if total == 0:
        return 0.0, (failed, total)
    return failed / total, (failed, total)


def re_question_rate(records: list[dict]) -> tuple[float, tuple[int, int]]:
    """再質問率と (retried_threads, total_threads) を返す。"""
    exits_by_corr: dict[str, int] = defaultdict(int)
    for record in records:
        event_type = record.get("event_type")
        if event_type not in _TASK_EXIT:
            continue
        corr = record.get("correlation_id")
        if not isinstance(corr, str):
            continue
        if not corr.startswith("slack:"):
            continue
        if corr.startswith("issuesmith:"):
            continue
        exits_by_corr[corr] += 1

    total_threads = len(exits_by_corr)
    retried = sum(1 for count in exits_by_corr.values() if count >= 2)
    if total_threads == 0:
        return 0.0, (retried, total_threads)
    return retried / total_threads, (retried, total_threads)


def memory_recall_rate(records: list[dict]) -> tuple[float, tuple[int, int]]:
    """メモリリコール率と (hit, total) を返す。hit + miss == 0 のとき rate は 0.0。"""
    hit = sum(1 for r in records if r.get("event_type") in _MEMORY_HIT)
    miss = sum(1 for r in records if r.get("event_type") in _MEMORY_MISS)
    total = hit + miss
    if total == 0:
        return 0.0, (hit, total)
    return hit / total, (hit, total)


def task_completion_time_ms(records: list[dict]) -> float | None:
    """タスク完了時間の平均（ミリ秒）。有効レコードが 0 件のとき None。"""
    elapsed_secs: list[float] = []
    for record in records:
        if record.get("event_type") not in _TASK_COMPLETION:
            continue
        elapsed = record.get("elapsed_sec")
        if elapsed is None:
            continue
        elapsed_secs.append(float(elapsed))
    if not elapsed_secs:
        return None
    return sum(elapsed_secs) / len(elapsed_secs) * 1000.0

