"""Conversation-layer session ledger and turn log (#3317).

Keyed by opaque conversation ID. No engine SDK dependency.
Resume eligibility is injected via configure_resume_check.
Storage is injected via ConversationConfig.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from mltgnt.config import ConversationConfig

__all__ = [
    "SessionRecord",
    "append_turn",
    "configure",
    "configure_paths",
    "configure_resume_check",
    "gc_sessions",
    "invalidate_ledger",
    "invalidate_session",
    "latest_session",
    "load_turns",
    "lookup_ledger",
    "record_ledger_entry",
    "record_session",
    "resume_supported",
    "session_exists",
    "session_path",
    "sessions_dir",
    "storage_key",
]

_JST = timezone(timedelta(hours=9))

# Query to the engine layer (resume disabled when unset)
_ResumeCheck = Callable[[str], bool]
_resume_check: _ResumeCheck | None = None
_active_config: ConversationConfig | None = None
_sessions_dir_override: Path | None = None
_ledger_dir_override: Path | None = None


@dataclass(frozen=True)
class SessionRecord:
    """Session metadata held by the conversation layer (media/engine-agnostic)."""

    engine: str
    session_id: str
    created_at: datetime
    parent_session_id: str | None = None
    is_compacted: bool = False
    summary_tokens: int | None = None


def configure(config: ConversationConfig) -> None:
    """Inject storage paths from ConversationConfig."""
    global _active_config, _sessions_dir_override, _ledger_dir_override
    _active_config = config
    _sessions_dir_override = config.sessions_dir
    _ledger_dir_override = config.ledger_dir


def configure_resume_check(check: _ResumeCheck | None) -> None:
    """Inject resume eligibility check. Pass None to clear."""
    global _resume_check
    _resume_check = check


def configure_paths(
    *,
    sessions_dir: Path | None = None,
    ledger_dir: Path | None = None,
) -> None:
    """Inject storage paths (host compatibility path)."""
    global _sessions_dir_override, _ledger_dir_override
    if sessions_dir is not None:
        _sessions_dir_override = sessions_dir
    if ledger_dir is not None:
        _ledger_dir_override = ledger_dir


def sessions_dir() -> Path:
    if _sessions_dir_override is not None:
        return _sessions_dir_override
    if _active_config is not None:
        return _active_config.sessions_dir
    raise RuntimeError(
        "session_store is not configured; call mltgnt.conversation.configure() first"
    )


_sessions_dir = sessions_dir  # deprecated alias


def _ledger_dir() -> Path:
    if _ledger_dir_override is not None:
        return _ledger_dir_override
    if _active_config is not None:
        return _active_config.ledger_dir
    raise RuntimeError(
        "session_store is not configured; call mltgnt.conversation.configure() first"
    )


def storage_key(conversation_id: str) -> str:
    """conversation ID → filename key (path-safe opaque string)."""
    return conversation_id.replace(":", "-").replace("/", "_")


def session_path(conversation_id: str) -> Path:
    return sessions_dir() / f"{storage_key(conversation_id)}.jsonl"


def _ledger_path(key: str) -> Path:
    return _ledger_dir() / f"{key}.json"


def record_ledger_entry(key: str, engine: str, session_id: str) -> None:
    """Record to the ledger with a direct key (compat keys)."""
    _atomic_write_ledger(
        key,
        {
            "engine": engine,
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def record_session(conversation_id: str, engine: str, session_id: str) -> None:
    record_ledger_entry(storage_key(conversation_id), engine, session_id)


def latest_session(conversation_id: str) -> SessionRecord | None:
    return lookup_ledger(storage_key(conversation_id))


def lookup_ledger(key: str, *, max_age: timedelta | None = None) -> SessionRecord | None:
    path = _ledger_path(key)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    engine = data.get("engine")
    session_id = data.get("session_id")
    if not isinstance(engine, str) or not engine:
        return None
    if not isinstance(session_id, str) or not session_id:
        return None
    created_at = _resolve_created_at(data, path)
    if created_at is None:
        return None
    if max_age is not None and created_at + max_age <= datetime.now(timezone.utc):
        return None
    parent_session_id = data.get("parent_session_id")
    if parent_session_id is not None and not isinstance(parent_session_id, str):
        parent_session_id = None
    is_compacted = data.get("is_compacted", False)
    if not isinstance(is_compacted, bool):
        is_compacted = False
    summary_tokens = data.get("summary_tokens")
    if summary_tokens is not None and not isinstance(summary_tokens, int):
        summary_tokens = None
    return SessionRecord(
        engine=engine,
        session_id=session_id,
        created_at=created_at,
        parent_session_id=parent_session_id,
        is_compacted=is_compacted,
        summary_tokens=summary_tokens,
    )


def invalidate_session(conversation_id: str) -> bool:
    return invalidate_ledger(storage_key(conversation_id))


def invalidate_ledger(key: str) -> bool:
    path = _ledger_path(key)
    if not path.exists():
        return False
    path.unlink()
    return True


def gc_sessions(*, max_age: timedelta) -> int:
    store_dir = _ledger_dir()
    if not store_dir.is_dir():
        return 0
    now = datetime.now(timezone.utc)
    deleted = 0
    for path in store_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        created_at = _resolve_created_at(data if isinstance(data, dict) else {}, path)
        if created_at is None or created_at + max_age <= now:
            path.unlink(missing_ok=True)
            deleted += 1
    return deleted


def resume_supported(engine: str | None) -> bool:
    """Return resume eligibility via injected callback. False if unset or empty engine."""
    name = (engine or "").strip()
    if not name:
        return False
    check = _resume_check
    if check is None:
        return False
    return bool(check(name))


def append_turn(
    conversation_id: str,
    role: str,
    content: str,
    persona_id: str | None = None,
    *,
    persona: str | None = None,
    raw_result_path: str | None = None,
) -> None:
    """Append a turn. persona is a backward-compat alias."""
    del raw_result_path  # accepted for compatibility, not stored
    path = session_path(conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    persona_val = persona_id if persona_id is not None else persona
    entry: dict = {
        "kind": "turn",
        "role": role,
        "persona": persona_val,
        "content": content,
        "ts": datetime.now(_JST).isoformat(),
    }
    if entry["persona"] is None:
        del entry["persona"]
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(line)
        f.flush()


def load_turns(conversation_id: str) -> list[dict]:
    path = session_path(conversation_id)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def session_exists(conversation_id: str) -> bool:
    return session_path(conversation_id).exists()


def _atomic_write_ledger(key: str, payload: dict) -> None:
    store_dir = _ledger_dir()
    store_dir.mkdir(parents=True, exist_ok=True)
    target = _ledger_path(key)
    fd, tmp = tempfile.mkstemp(dir=store_dir, suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
        Path(tmp).replace(target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _resolve_created_at(data: dict, path: Path) -> datetime | None:
    raw = data.get("created_at")
    if isinstance(raw, str) and raw:
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            pass
        else:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None
