"""会話固定ペルソナの JSONL 永続化（#3317）。

会話 ID でキーする。インメモリ map の更新はコールバックで注入する。
保存先・TTL は ConversationConfig で注入する。
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from mltgnt.config import ConversationConfig

_log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_active_config: ConversationConfig | None = None
_store_path_override: Path | None = None

_OnSet = Callable[[str, str], None]
_on_set: _OnSet | None = None


def configure(config: ConversationConfig) -> None:
    global _active_config, _store_path_override
    _active_config = config
    _store_path_override = config.thread_persona_path


def configure_on_set(callback: _OnSet | None) -> None:
    """インメモリ map 更新用コールバックを注入する。"""
    global _on_set
    _on_set = callback


def _store_path() -> Path:
    if _store_path_override is not None:
        return _store_path_override
    if _active_config is not None:
        return _active_config.thread_persona_path
    raise RuntimeError(
        "thread_persona_store is not configured; call mltgnt.conversation.configure() first"
    )


def _ttl_days() -> int:
    if _active_config is not None:
        return int(_active_config.thread_persona_ttl_days)
    raw = os.environ.get("THREAD_PERSONA_TTL_DAYS", "30")
    try:
        return int(raw)
    except ValueError:
        return 30


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def _is_expired(ts: str, *, now: datetime | None = None) -> bool:
    parsed = _parse_ts(ts)
    if parsed is None:
        return True
    if now is None:
        now = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (now - parsed).days >= _ttl_days()


def _compact(entries: dict[str, tuple[str, str]]) -> None:
    path = _store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps({"key": key, "persona": persona, "ts": ts}, ensure_ascii=False)
            for key, (persona, ts) in entries.items()
        ]
        content = "\n".join(lines)
        if content:
            content += "\n"
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            os.write(fd, content.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except OSError:
        _log.warning("[thread_persona_store] compaction failed", exc_info=True)


def load() -> dict[str, str]:
    path = _store_path()
    if not path.exists():
        return {}

    result: dict[str, str] = {}
    compact_entries: dict[str, tuple[str, str]] = {}

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        _log.warning("[thread_persona_store] load failed", exc_info=True)
        return {}

    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            _log.warning("[thread_persona_store] invalid JSON at line %d", line_no)
            continue

        key = entry.get("key")
        persona = entry.get("persona")
        ts = entry.get("ts")
        if not isinstance(key, str) or not isinstance(persona, str) or not isinstance(ts, str):
            _log.warning("[thread_persona_store] invalid entry at line %d", line_no)
            continue
        if _is_expired(ts):
            continue

        result[key] = persona
        compact_entries[key] = (persona, ts)

    _compact(compact_entries)
    return result


def set_persona(conversation_id: str, persona: str) -> None:
    """会話 ID にペルソナを固定し、JSONL に追記する。"""
    key = conversation_id
    cb = _on_set
    if cb is not None:
        try:
            cb(key, persona)
        except Exception:
            _log.warning("[thread_persona_store] on_set callback failed", exc_info=True)

    ts = datetime.now(timezone.utc).isoformat()
    entry = {"key": key, "persona": persona, "ts": ts}
    path = _store_path()

    with _LOCK:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            _log.warning("[thread_persona_store] append failed: key=%s", key, exc_info=True)
