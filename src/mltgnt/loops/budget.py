"""mltgnt.loops.budget — JST 日次共有 LLM 予算の atomic reserve。"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

_TZ = ZoneInfo("Asia/Tokyo")
_LOCK_TIMEOUT_SEC = 5.0
_LOCK_STALE_SEC = 30.0


@dataclass(frozen=True)
class BudgetReserveResult:
    allowed: bool
    loop_count: int
    day_count: int
    loop_limit: int
    day_limit: int
    reason: str = ""


def jst_date_str(now: datetime | None = None) -> str:
    current = now or datetime.now(_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=_TZ)
    return current.astimezone(_TZ).strftime("%Y-%m-%d")


def budget_dir(state_dir: Path) -> Path:
    return state_dir / "llm-budget"


def budget_path(state_dir: Path, day: str) -> Path:
    return budget_dir(state_dir) / f"{day}.json"


def _lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


def _acquire_lock(lock: Path, *, timeout_sec: float = _LOCK_TIMEOUT_SEC) -> None:
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("ascii"))
            finally:
                os.close(fd)
            return
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except OSError:
                age = 0.0
            if age >= _LOCK_STALE_SEC:
                try:
                    lock.unlink(missing_ok=True)
                except OSError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"budget lock timeout: {lock}")
            time.sleep(0.02)


def _release_lock(lock: Path) -> None:
    try:
        lock.unlink(missing_ok=True)
    except OSError:
        pass


def _read_day_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return 0
    if not isinstance(data, dict):
        return 0
    raw = data.get("count", 0)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return 0
    return raw


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_name = f.name
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def reserve_llm_call(
    state_dir: Path,
    *,
    loop_id: str,
    loop_count: int,
    loop_limit: int,
    day_limit: int,
    budget_override: bool = False,
    now: datetime | None = None,
) -> BudgetReserveResult:
    """物理 LLM 試行の直前に1回予約する。

    override 時もカウンタは進めるが、両上限による拒否はしない。
    上限 0 は「呼び出し禁止」（override 除く）。
    """
    if loop_count < 0:
        raise ValueError("loop_count must be non-negative")
    if loop_limit < 0 or day_limit < 0:
        raise ValueError("budget limits must be non-negative")

    day = jst_date_str(now)
    path = budget_path(state_dir, day)
    lock = _lock_path(path)
    _acquire_lock(lock)
    try:
        day_count = _read_day_count(path)
        next_loop = loop_count + 1
        next_day = day_count + 1

        if not budget_override:
            if loop_limit == 0 or next_loop > loop_limit:
                return BudgetReserveResult(
                    allowed=False,
                    loop_count=loop_count,
                    day_count=day_count,
                    loop_limit=loop_limit,
                    day_limit=day_limit,
                    reason="loop_budget_exceeded",
                )
            if day_limit == 0 or next_day > day_limit:
                return BudgetReserveResult(
                    allowed=False,
                    loop_count=loop_count,
                    day_count=day_count,
                    loop_limit=loop_limit,
                    day_limit=day_limit,
                    reason="day_budget_exceeded",
                )

        _atomic_write(
            path,
            {
                "date": day,
                "count": next_day,
                "last_loop_id": loop_id,
            },
        )
        return BudgetReserveResult(
            allowed=True,
            loop_count=next_loop,
            day_count=next_day,
            loop_limit=loop_limit,
            day_limit=day_limit,
        )
    finally:
        _release_lock(lock)
