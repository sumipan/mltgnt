"""KPI 算出ロジック。"""
from __future__ import annotations

from collections import defaultdict

_TASK_COMPLETE = frozenset({"task_complete"})
_TASK_FAILED = frozenset({"task_failed"})
_TASK_EXIT = _TASK_COMPLETE | _TASK_FAILED


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

