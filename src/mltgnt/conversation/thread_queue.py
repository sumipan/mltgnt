"""会話層の待機列（#3317）。

会話 ID でキーし、drain は TurnInput を返すだけ。Slack API / dispatch は持たない。
保存先は ConversationConfig（または THREAD_QUEUE_DIR 上書き）で注入する。
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from mltgnt.conversation.types import HistoryMessage, TurnInput

if TYPE_CHECKING:
    from mltgnt.config import ConversationConfig

_log = logging.getLogger(__name__)

AdmitResult = namedtuple("AdmitResult", ["proceed", "queued", "status"])

# テストの monkeypatch 互換（configure 前のフォールバックにも使う）
THREAD_QUEUE_DIR: Path | None = None

_CANCEL_WORDS = frozenset({"キャンセル", "止めて", "cancel", "stop"})

_locks_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}
_active_config: ConversationConfig | None = None

_COMPOSITE_HEADER = (
    "処理中に以下の発言がありました。これらを踏まえて対応してください。"
)
_COMPOSITE_CANCEL_SUFFIX = (
    "※ 中止指示が含まれています。現在の作業を中止し、中止した旨を報告してください。"
)


def configure(config: ConversationConfig) -> None:
    """待機列の保存先・閾値を注入する。"""
    global _active_config, THREAD_QUEUE_DIR
    _active_config = config
    THREAD_QUEUE_DIR = config.queue_dir


def _thread_queue_dir() -> Path:
    if THREAD_QUEUE_DIR is not None:
        return THREAD_QUEUE_DIR
    if _active_config is not None:
        return _active_config.queue_dir
    raise RuntimeError(
        "thread_queue is not configured; call mltgnt.conversation.configure() first"
    )


def _active_thread_queue_dir() -> Path:
    return _thread_queue_dir()


def _thread_queue_config() -> dict[str, int]:
    if _active_config is not None:
        return {
            "stale_after_sec": int(_active_config.stale_after_sec),
            "max_queued": int(_active_config.max_queued),
            "cleanup_ttl_days": int(_active_config.cleanup_ttl_days),
        }
    return {
        "stale_after_sec": 3600,
        "max_queued": 20,
        "cleanup_ttl_days": 14,
    }


def storage_key(conversation_id: str) -> str:
    """会話 ID → 状態ディレクトリ名。"""
    return conversation_id.replace(":", "-").replace("/", "_")


def _get_lock(thread_key: str) -> threading.Lock:
    with _locks_guard:
        if thread_key not in _locks:
            _locks[thread_key] = threading.Lock()
        return _locks[thread_key]


def _thread_dir(thread_key: str) -> Path:
    return _active_thread_queue_dir() / thread_key


def _state_path(thread_key: str) -> Path:
    return _thread_dir(thread_key) / "state.json"


def _inbox_dir(thread_key: str) -> Path:
    return _thread_dir(thread_key) / "inbox"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".inbox-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_state(thread_key: str) -> dict:
    path = _state_path(thread_key)
    if not path.is_file():
        return {"status": "idle", "started_at": None, "current_uuid": None}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"status": "idle", "started_at": None, "current_uuid": None}
    if not isinstance(raw, dict):
        return {"status": "idle", "started_at": None, "current_uuid": None}
    status = raw.get("status", "idle")
    if status not in ("idle", "running"):
        status = "idle"
    return {
        "status": status,
        "started_at": raw.get("started_at"),
        "current_uuid": raw.get("current_uuid"),
    }


def _write_state(thread_key: str, state: dict) -> None:
    _atomic_write_json(_state_path(thread_key), state)


def _count_inbox(thread_key: str) -> int:
    inbox = _inbox_dir(thread_key)
    if not inbox.is_dir():
        return 0
    return sum(1 for p in inbox.iterdir() if p.is_file() and p.suffix == ".json")


def _is_cancel_instruction(instruction: str) -> bool:
    return instruction.strip() in _CANCEL_WORDS


def _is_stale(state: dict, stale_after_sec: int) -> bool:
    if state.get("status") != "running":
        return False
    started_at = state.get("started_at")
    if not started_at:
        return False
    try:
        started = datetime.fromisoformat(str(started_at))
    except ValueError:
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - started > timedelta(seconds=stale_after_sec)


def _write_inbox_entry(
    thread_key: str,
    *,
    conversation_id: str,
    message_ts: str,
    author: str,
    instruction: str,
    kind: str,
) -> None:
    payload = {
        "kind": kind,
        "ts": message_ts,
        "conversation_id": conversation_id,
        "user": author,
        "text": instruction,
        "received_at": _utc_now_iso(),
    }
    _atomic_write_json(_inbox_dir(thread_key) / f"{message_ts}.json", payload)


def admit(
    conversation_id: str,
    instruction: str,
    *,
    message_ts: str,
    author: str = "",
) -> AdmitResult:
    """idle→accepted / running→queued / 上限超過→rejected。リアクションは付けない。"""
    thread_key = storage_key(conversation_id)
    lock = _get_lock(thread_key)
    with lock:
        cfg = _thread_queue_config()
        state = _read_state(thread_key)

        if _is_stale(state, cfg["stale_after_sec"]):
            _log.warning(
                "thread_queue stale reset thread_key=%s started_at=%s",
                thread_key,
                state.get("started_at"),
            )
            state = {"status": "idle", "started_at": None, "current_uuid": None}
            _write_state(thread_key, state)

        if state["status"] == "idle":
            state = {
                "status": "running",
                "started_at": _utc_now_iso(),
                "current_uuid": state.get("current_uuid"),
            }
            _write_state(thread_key, state)
            return AdmitResult(proceed=True, queued=False, status="accepted")

        queued_count = _count_inbox(thread_key)
        if queued_count >= cfg["max_queued"]:
            return AdmitResult(proceed=False, queued=False, status="rejected")

        kind = "cancel" if _is_cancel_instruction(instruction) else "message"
        _write_inbox_entry(
            thread_key,
            conversation_id=conversation_id,
            message_ts=message_ts,
            author=author,
            instruction=instruction,
            kind=kind,
        )
        return AdmitResult(proceed=False, queued=True, status="queued")


def record_job(conversation_id: str, uuid: str) -> None:
    thread_key = storage_key(conversation_id)
    lock = _get_lock(thread_key)
    with lock:
        state = _read_state(thread_key)
        state["current_uuid"] = uuid
        _write_state(thread_key, state)


def record_job_by_key(thread_key: str, uuid: str) -> None:
    """storage_key 直指定（旧 make_thread_key 互換）。"""
    lock = _get_lock(thread_key)
    with lock:
        state = _read_state(thread_key)
        state["current_uuid"] = uuid
        _write_state(thread_key, state)


def _read_inbox_entries(thread_key: str) -> list[dict]:
    inbox = _inbox_dir(thread_key)
    if not inbox.is_dir():
        return []
    entries: list[dict] = []
    for path in sorted(inbox.iterdir(), key=lambda p: p.name):
        if not path.is_file() or path.suffix != ".json":
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(raw, dict):
            entries.append(raw)
    return entries


def _clear_inbox(thread_key: str) -> None:
    inbox = _inbox_dir(thread_key)
    if not inbox.is_dir():
        return
    for path in inbox.iterdir():
        if path.is_file():
            path.unlink()


def finish_turn(thread_key: str) -> list[dict] | None:
    """ターン完了時に inbox を drain。空なら idle にして None。"""
    lock = _get_lock(thread_key)
    with lock:
        entries = _read_inbox_entries(thread_key)
        if not entries:
            _write_state(
                thread_key,
                {"status": "idle", "started_at": None, "current_uuid": None},
            )
            return None

        _clear_inbox(thread_key)
        _write_state(
            thread_key,
            {
                "status": "running",
                "started_at": _utc_now_iso(),
                "current_uuid": None,
            },
        )
        return entries


def build_composite_instruction(entries: list[dict]) -> str:
    lines = [_COMPOSITE_HEADER, ""]
    for index, entry in enumerate(entries, 1):
        lines.append(f"[{index}] {entry.get('text', '')}")
    if any(entry.get("kind") == "cancel" for entry in entries):
        lines.extend(["", _COMPOSITE_CANCEL_SUFFIX])
    return "\n".join(lines)


def drain_to_turn_input(
    conversation_id: str,
    *,
    persona_id: str | None = None,
) -> TurnInput | None:
    """inbox を drain し、次ターンの TurnInput を返す。空なら None。起動はしない。"""
    thread_key = storage_key(conversation_id)
    entries = finish_turn(thread_key)
    if entries is None:
        return None
    text = build_composite_instruction(entries)
    history = tuple(
        HistoryMessage(role="user", text=str(entry.get("text") or ""))
        for entry in entries
        if (entry.get("text") or "").strip()
    )
    return TurnInput(
        conversation_id=conversation_id,
        text=text,
        history=history,
        persona_id=persona_id,
    )


def drained_entries_meta(entries: list[dict]) -> dict:
    """入口計測用のメタ（queued_count / earliest received_at）。"""
    best: datetime | None = None
    for entry in entries:
        raw = entry.get("received_at")
        if not raw:
            continue
        try:
            parsed = datetime.fromisoformat(str(raw))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        if best is None or parsed < best:
            best = parsed
    return {
        "queued_count": len(entries),
        "earliest_received_at": best.isoformat() if best else None,
        "entries": entries,
    }
