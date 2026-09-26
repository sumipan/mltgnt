"""mltgnt.memory.archive — move old episodes to monthly archive files.

``<persona_dir>/episodes.jsonl`` lines older than ``now - keep_days`` are
appended to ``<persona_dir>/episodes/YYYY-MM.jsonl``; the remaining lines
replace ``episodes.jsonl`` through a tmp file and ``os.replace``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

__all__ = ["archive_episodes"]


def _parse_timestamp(line: str) -> datetime | None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("timestamp")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw.strip())
    except ValueError:
        return None


def archive_episodes(persona_dir: Path, *, now: datetime, keep_days: int = 90) -> int:
    """Archive episodes older than ``keep_days``. Returns the number of lines moved.

    Lines whose timestamp cannot be parsed stay in ``episodes.jsonl``.
    """
    persona_dir = Path(persona_dir)
    path = persona_dir / "episodes.jsonl"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return 0
    cutoff = now - timedelta(days=keep_days)
    keep: list[str] = []
    by_month: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        ts = _parse_timestamp(line)
        if ts is not None and ts.tzinfo is not None and cutoff.tzinfo is None:
            ts = ts.replace(tzinfo=None)
        elif ts is not None and ts.tzinfo is None and cutoff.tzinfo is not None:
            ts = ts.replace(tzinfo=cutoff.tzinfo)
        if ts is not None and ts < cutoff:
            by_month.setdefault(ts.strftime("%Y-%m"), []).append(line)
        else:
            keep.append(line)
    moved = sum(len(lines) for lines in by_month.values())
    if not moved:
        return 0
    archive_dir = persona_dir / "episodes"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for month, lines in sorted(by_month.items()):
        with (archive_dir / f"{month}.jsonl").open("a", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in lines))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(line + "\n" for line in keep), encoding="utf-8")
    os.replace(tmp, path)
    return moved
