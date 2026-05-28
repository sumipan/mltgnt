"""audit.jsonl ストリームパーサー。"""
from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_JST = ZoneInfo("Asia/Tokyo")


def iter_audit_records(audit_path: Path) -> Iterator[dict]:
    """audit.jsonl を 1 行ずつ読み、パース可能なレコードのみ yield する。"""
    with audit_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def record_date(record: dict) -> date | None:
    """レコードの timestamp から Asia/Tokyo の日付を返す。取得不能なら None。"""
    raw = record.get("timestamp")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_JST)
    return dt.astimezone(_JST).date()


def filter_records_by_period(
    records: Iterator[dict],
    *,
    since: date | None,
    until: date | None,
) -> list[dict]:
    """since/until（含む）で日付フィルタ。timestamp 欠落行は除外。"""
    out: list[dict] = []
    for record in records:
        d = record_date(record)
        if d is None:
            continue
        if since is not None and d < since:
            continue
        if until is not None and d > until:
            continue
        out.append(record)
    return out


def parse_date_arg(value: str) -> date:
    return date.fromisoformat(value)


def period_bounds(records: list[dict]) -> tuple[str, str]:
    """フィルタ済みレコードから期間表示用文字列を返す。"""
    dates: list[date] = []
    for record in records:
        d = record_date(record)
        if d is not None:
            dates.append(d)
    if not dates:
        return "—", "—"
    start, end = min(dates), max(dates)
    return start.isoformat(), end.isoformat()
