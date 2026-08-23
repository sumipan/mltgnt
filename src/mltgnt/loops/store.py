"""mltgnt.loops.store — state / events / inbox の永続化。"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from mltgnt.bridges.files_adapter import md_read, md_write
from mltgnt.loops.models import LoopState, SCHEMA_VERSION, state_from_json, state_to_json

logger = logging.getLogger("mltgnt.loops.store")

_TZ = ZoneInfo("Asia/Tokyo")
_OMISSION_MARKER = "\n\n…\n\n"


@dataclass(frozen=True)
class InboxMessage:
    kind: str
    message_id: str
    question_id: str
    text: str
    received_at: str
    filename: str


def _now_iso() -> str:
    return datetime.now(_TZ).isoformat()


def loop_state_dir(state_dir: Path, loop_id: str) -> Path:
    return state_dir / loop_id


def deliverable_path(state_dir: Path, loop_id: str) -> Path:
    """正規成果物 `deliverable.md` の絶対パスを返す。"""
    return (loop_state_dir(state_dir, loop_id) / "deliverable.md").resolve()


def _read_deliverable_text(state_dir: Path, loop_id: str) -> str:
    path = deliverable_path(state_dir, loop_id)
    if not path.is_file():
        return ""
    md = md_read(path.name, repo_root=path.parent)
    content = getattr(md, "content", md)
    return content if isinstance(content, str) else str(content)


def initialize_deliverable(state_dir: Path, loop_id: str, body: str) -> Path:
    """Objective 本文で deliverable.md を初期化し、保存先絶対パスを返す。"""
    path = deliverable_path(state_dir, loop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = body if body.endswith("\n") else body + "\n"
    md_write(path.name, content, repo_root=path.parent)
    return path


def read_deliverable_excerpt(state_dir: Path, loop_id: str, max_chars: int) -> str:
    """deliverable 全文が上限以下なら全文、超過時は先頭・省略マーカー・末尾。"""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    text = _read_deliverable_text(state_dir, loop_id)
    if len(text) <= max_chars:
        return text
    marker = _OMISSION_MARKER
    if len(marker) >= max_chars:
        return text[:max_chars]
    remaining = max_chars - len(marker)
    head_len = remaining // 2
    tail_len = remaining - head_len
    if tail_len <= 0:
        return text[:max_chars]
    return text[:head_len] + marker + text[-tail_len:]


def deliverable_snapshot(state_dir: Path, loop_id: str) -> dict[str, Any]:
    """成果物スナップショット（path / chars / bytes）を返す。"""
    path = deliverable_path(state_dir, loop_id)
    text = _read_deliverable_text(state_dir, loop_id)
    return {
        "path": str(path),
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
    }


def atomic_write_json(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def save_state(state_dir: Path, state: LoopState) -> None:
    path = loop_state_dir(state_dir, state.loop_id) / "state.json"
    atomic_write_json(path, state_to_json(state))


def load_state(state_dir: Path, loop_id: str) -> LoopState | None:
    path = loop_state_dir(state_dir, loop_id) / "state.json"
    if not path.is_file():
        return None
    try:
        return state_from_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"corrupt state for {loop_id}: {exc}") from exc


def append_event(state_dir: Path, loop_id: str, event: str, data: dict[str, Any], *, iteration: int) -> None:
    path = loop_state_dir(state_dir, loop_id) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": _now_iso(),
        "loop_id": loop_id,
        "iteration": iteration,
        "event": event,
        "data": data,
    }
    # LLMResult / dataclass 等が混ざっても状態遷移を止めない
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_events(state_dir: Path, loop_id: str) -> list[dict[str, Any]]:
    path = loop_state_dir(state_dir, loop_id) / "events.jsonl"
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    events: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            events.append(json.loads(stripped))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                logger.warning("ignoring corrupt trailing event line for %s", loop_id)
            else:
                raise
    return events


def list_restorable_loops(state_dir: Path) -> list[str]:
    if not state_dir.is_dir():
        return []
    result: list[str] = []
    for child in state_dir.iterdir():
        if child.is_dir() and (child / "state.json").is_file():
            result.append(child.name)
    return sorted(result)


def archive_terminal_state(state_dir: Path, loop_id: str) -> Path | None:
    """終端 state ディレクトリを state_dir/archive/ へ atomic rename する。"""
    src = loop_state_dir(state_dir, loop_id)
    if not src.is_dir():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = state_dir / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    dest = archive_root / f"{loop_id}.archived-{stamp}"
    os.replace(src, dest)
    return dest


def mark_state_corrupt(state_dir: Path, loop_id: str, reason: str) -> None:
    corrupt_dir = state_dir / loop_id / "corrupt"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    (corrupt_dir / "reason.txt").write_text(reason, encoding="utf-8")


def _inbox_dir(state_dir: Path, loop_id: str) -> Path:
    return loop_state_dir(state_dir, loop_id) / "inbox"


def _consumed_dir(state_dir: Path, loop_id: str) -> Path:
    return _inbox_dir(state_dir, loop_id) / "consumed"


def list_inbox_messages(state_dir: Path, loop_id: str) -> list[InboxMessage]:
    inbox = _inbox_dir(state_dir, loop_id)
    if not inbox.is_dir():
        return []
    messages: list[InboxMessage] = []
    for path in sorted(inbox.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("skipping invalid inbox JSON: %s", path)
            continue
        for field in ("kind", "message_id", "question_id", "text", "received_at"):
            if field not in data:
                logger.warning("skipping inbox missing %s: %s", field, path)
                break
        else:
            if data["kind"] not in ("answer", "cancel", "comment"):
                logger.warning("skipping inbox invalid kind: %s", path)
                continue
            if not all(isinstance(data[field], str) for field in (
                "kind", "message_id", "question_id", "text", "received_at"
            )):
                logger.warning("skipping inbox with non-string field: %s", path)
                continue
            messages.append(
                InboxMessage(
                    kind=str(data["kind"]),
                    message_id=str(data["message_id"]),
                    question_id=str(data["question_id"]),
                    text=str(data["text"]),
                    received_at=str(data["received_at"]),
                    filename=path.name,
                )
            )
    return messages


def consume_inbox_message(state_dir: Path, loop_id: str, filename: str) -> None:
    src = _inbox_dir(state_dir, loop_id) / filename
    dst = _consumed_dir(state_dir, loop_id) / filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)


def list_consumed_message_ids(state_dir: Path, loop_id: str) -> set[str]:
    consumed = _consumed_dir(state_dir, loop_id)
    if not consumed.is_dir():
        return set()
    ids: set[str] = set()
    for path in consumed.iterdir():
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ids.add(str(data["message_id"]))
        except (json.JSONDecodeError, KeyError):
            pass
    return ids
