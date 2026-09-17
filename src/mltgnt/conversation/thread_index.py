"""Conversation-layer thread index (#3317).

Keyed by conversation ID. No external engine SDK dependency.
Storage is injected via ConversationConfig.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mltgnt.config import ConversationConfig

_log = logging.getLogger(__name__)

_active_config: ConversationConfig | None = None


def configure(config: ConversationConfig) -> None:
    global _active_config
    _active_config = config


def _thread_index_dir() -> Path:
    if _active_config is None:
        raise RuntimeError(
            "thread_index is not configured; call mltgnt.conversation.configure() first"
        )
    return _active_config.thread_index_dir


def _posts_dir() -> Path:
    if _active_config is None:
        raise RuntimeError(
            "thread_index is not configured; call mltgnt.conversation.configure() first"
        )
    if _active_config.posts_dir is not None:
        return _active_config.posts_dir
    return _active_config.thread_index_dir.parent / "posts"


def storage_key(conversation_id: str) -> str:
    return conversation_id.replace(":", "-").replace("/", "_")


def append_entry(
    conversation_id: str,
    posted_ts: str,
    uid: str,
    result_path: str,
    order_path: str,
) -> None:
    index_dir = _thread_index_dir()
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"{storage_key(conversation_id)}.jsonl"
    entry = {
        "posted_ts": posted_ts,
        "uid": uid,
        "result_path": result_path,
        "order_path": order_path,
    }
    try:
        with index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        _log.warning(
            "[thread_index] append_entry failed: conversation_id=%s",
            conversation_id,
            exc_info=True,
        )


def _lookup_in_file(index_path: Path, posted_ts: str) -> str | None:
    try:
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("posted_ts") == posted_ts:
                    result_path = entry.get("result_path")
                    return result_path if isinstance(result_path, str) else None
            except json.JSONDecodeError:
                continue
    except OSError:
        return None
    return None


def _write_post_content(uid: str, content: str) -> str:
    """Persist reply body and return a relative-path-like reference string."""
    posts = _posts_dir()
    posts.mkdir(parents=True, exist_ok=True)
    target = posts / f"{uid}.md"
    target.write_text(content, encoding="utf-8")
    return str(target)


def register_bot_post(
    conversation_id: str,
    posted_ts: str,
    uid: str,
    *,
    content: str | None = None,
    result_path: str | None = None,
    order_path: str = "",
) -> None:
    if result_path is None and content is not None:
        result_path = _write_post_content(uid, content)
    append_entry(conversation_id, posted_ts, uid, result_path or "", order_path)


def lookup_result_path(
    conversation_id: str,
    posted_ts: str,
    *,
    legacy_dirs: tuple[Path, ...] = (),
) -> str | None:
    filename = f"{storage_key(conversation_id)}.jsonl"
    for base in (_thread_index_dir(), *legacy_dirs):
        index_path = base / filename
        if not index_path.exists():
            continue
        result = _lookup_in_file(index_path, posted_ts)
        if result is not None:
            return result
    return None
