from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _hash_offset(job_id: str, local_date: str, salt: str, span: int) -> int:
    if span <= 0:
        return 0
    payload = f"{job_id}|{local_date}|{salt}".encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    return int(h[:12], 16) % span


class SchedulePaths:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.done_dir = state_dir / "done"
        self.planned_dir = state_dir / "planned"
        self.missed_dir = state_dir / "missed"
        self.failed_dir = state_dir / "failed"
        self.skipped_dir = state_dir / "skipped"
        self.interval_dir = state_dir / "interval"

    def done_path(self, job_id: str, d: date) -> Path:
        return self.done_dir / f"{job_id}_{d.isoformat()}.done"

    def planned_path(self, job_id: str, d: date) -> Path:
        return self.planned_dir / f"{job_id}_{d.isoformat()}.json"

    def missed_path(self, job_id: str, d: date) -> Path:
        return self.missed_dir / f"{job_id}_{d.isoformat()}.flag"

    def failed_path(self, job_id: str, d: date) -> Path:
        return self.failed_dir / f"{job_id}_{d.isoformat()}.failed"

    def skipped_path(self, job_id: str, d: date) -> Path:
        return self.skipped_dir / f"{job_id}_{d.isoformat()}.skipped"

    def interval_last_fired_path(self, job_id: str) -> Path:
        return self.interval_dir / f"{job_id}.last"

    def read_interval_last_fired(self, job_id: str) -> Optional[datetime]:
        p = self.interval_last_fired_path(job_id)
        if not p.exists():
            return None
        try:
            return datetime.fromisoformat(p.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            return None

    def write_interval_last_fired(self, job_id: str, dt: datetime) -> None:
        atomic_write_text(self.interval_last_fired_path(job_id), dt.isoformat())

    def load_all_interval_last_fired(self) -> dict[str, datetime]:
        result: dict[str, datetime] = {}
        if not self.interval_dir.exists():
            return result
        for p in self.interval_dir.glob("*.last"):
            job_id = p.stem
            try:
                result[job_id] = datetime.fromisoformat(
                    p.read_text(encoding="utf-8").strip()
                )
            except (ValueError, OSError):
                continue
        return result

    def prune(self, today: date, keep_days: int = 30) -> int:
        """Delete dated state files older than ``today - keep_days``.

        Covers the done / planned / missed / failed / skipped directories
        (``<job_id>_<YYYY-MM-DD>.<ext>``). Files whose date cannot be parsed
        are kept; ``interval_dir`` is never touched. Returns the number of
        files deleted.
        """
        if keep_days < 1:
            raise ValueError(f"keep_days must be >= 1, got {keep_days}")
        cutoff = today - timedelta(days=keep_days)
        removed = 0
        for d in (
            self.done_dir,
            self.planned_dir,
            self.missed_dir,
            self.failed_dir,
            self.skipped_dir,
        ):
            if not d.is_dir():
                continue
            for p in d.iterdir():
                _, sep, date_part = p.stem.rpartition("_")
                if not sep:
                    continue
                try:
                    file_date = date.fromisoformat(date_part)
                except ValueError:
                    continue
                if file_date >= cutoff:
                    continue
                try:
                    if not p.is_file():
                        continue
                    p.unlink(missing_ok=True)
                except OSError:
                    continue
                removed += 1
        return removed
