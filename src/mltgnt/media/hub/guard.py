"""Pure guard checks for hub message posting (no I/O)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from mltgnt.interfaces.hub import HubMessage

__all__ = ["GuardConfig", "GuardVerdict", "check"]


@dataclass(frozen=True)
class GuardConfig:
    hop_limit: int = 3
    max_agent_run: int = 4
    min_interval_sec: float = 5.0
    room_daily_limit: int = 200


@dataclass(frozen=True)
class GuardVerdict:
    ok: bool
    reason: str | None = None


def _is_human_row(row: Mapping[str, Any]) -> bool:
    sender = row.get("sender")
    if isinstance(sender, dict) and "persona" in sender:
        return sender.get("persona") is None
    author = row.get("author")
    if isinstance(author, str) and author.endswith("/user"):
        return True
    if isinstance(author, str) and "/" in author:
        return author.split("/", 1)[1] == "user"
    return False


def _persona_run_after_human(thread_rows: Sequence[Mapping[str, Any]]) -> int:
    """Count persona ``message`` rows after the last human row in chronological order."""
    last_human = -1
    for idx, row in enumerate(thread_rows):
        if row.get("kind") == "message" and _is_human_row(row):
            last_human = idx
    run = 0
    for row in thread_rows[last_human + 1 :]:
        if row.get("kind") != "message":
            continue
        if _is_human_row(row):
            run = 0
        else:
            run += 1
    return run


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _same_address(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return left.get("host") == right.get("host") and left.get("persona") == right.get("persona")


def check(
    message: HubMessage,
    thread_rows: Sequence[Mapping[str, Any]],
    room_today_count: int,
    now: datetime,
    *,
    config: GuardConfig | None = None,
) -> GuardVerdict:
    cfg = config if config is not None else GuardConfig()
    if message.kind != "message":
        return GuardVerdict(ok=True)

    if message.hop > cfg.hop_limit:
        return GuardVerdict(ok=False, reason="hop_exceeded")

    if room_today_count >= cfg.room_daily_limit:
        return GuardVerdict(ok=False, reason="room_daily_limit")

    if message.sender.persona is not None:
        run = _persona_run_after_human(thread_rows)
        if run >= cfg.max_agent_run:
            return GuardVerdict(ok=False, reason="agent_run_exceeded")

    sender_map = {"host": message.sender.host}
    if message.sender.persona is not None:
        sender_map["persona"] = message.sender.persona
    to_map = None
    if message.to is not None:
        to_map = {"host": message.to.host}
        if message.to.persona is not None:
            to_map["persona"] = message.to.persona

    for row in reversed(thread_rows):
        if row.get("kind") != "message":
            continue
        if not _same_address(row.get("sender") if isinstance(row.get("sender"), dict) else None, sender_map):
            continue
        if not _same_address(row.get("to") if isinstance(row.get("to"), dict) else None, to_map):
            continue
        prev_ts = _parse_ts(row.get("ts"))
        if prev_ts is None:
            break
        delta = (now - prev_ts).total_seconds()
        if delta < cfg.min_interval_sec:
            return GuardVerdict(ok=False, reason="too_fast")
        break

    return GuardVerdict(ok=True)
