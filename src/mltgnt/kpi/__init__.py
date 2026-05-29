"""KPI 計測 — audit.jsonl から応答失敗率・再質問率を集計する。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from mltgnt.kpi._metrics import re_question_rate, response_failure_rate
from mltgnt.kpi._parser import filter_records_by_period, iter_audit_records, period_bounds


@dataclass
class KPIReport:
    period_start: str
    period_end: str
    response_failure_rate: float
    response_failure_detail: tuple[int, int]
    re_question_rate: float
    re_question_detail: tuple[int, int]
    skill_resolution_rate: float | None


def compute_kpis(
    audit_path: Path | str,
    *,
    since: date | str | None = None,
    until: date | str | None = None,
) -> KPIReport:
    """audit.jsonl を読み KPIReport を返す。ファイル不存在時は FileNotFoundError。"""
    path = Path(audit_path)
    if not path.is_file():
        raise FileNotFoundError(f"audit file not found: {path}")

    since_date = _coerce_date(since)
    until_date = _coerce_date(until)

    records = filter_records_by_period(
        iter_audit_records(path),
        since=since_date,
        until=until_date,
    )

    rfr, rfd = response_failure_rate(records)
    rqr, rqd = re_question_rate(records)
    p_start, p_end = period_bounds(records)

    return KPIReport(
        period_start=p_start,
        period_end=p_end,
        response_failure_rate=rfr,
        response_failure_detail=rfd,
        re_question_rate=rqr,
        re_question_detail=rqd,
        skill_resolution_rate=None,
    )


def _coerce_date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)
