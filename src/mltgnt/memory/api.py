"""
mltgnt.memory.api — パス解決・ロック・追記・読み取り（CRUD 操作）。
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from mltgnt.config import MemoryConfig

from mltgnt.memory._format import (
    MemoryEntry,
    assemble_entries_text,
    parse_jsonl,
    serialize_entry,
)

LlmCall = Callable[[str], str]

_log = logging.getLogger(__name__)

MEMORY_DEDUPE_SCAN_BYTES = 32 * 1024
MEMORY_DEDUPE_SCAN_LINES = 200
MEMORY_CORRUPT_THRESHOLD_BYTES = 10


def _resolve_memory_dir(config: "MemoryConfig") -> Path:
    if config.chat_memory_dir is not None:
        return config.chat_memory_dir
    return config.chat_dir / "memory"


def memory_file_path(config: "MemoryConfig", persona_stem: str) -> Path:
    """`_resolve_memory_dir(config) / f\"{persona_stem}.jsonl\"`"""
    return _resolve_memory_dir(config) / f"{persona_stem}.jsonl"


def normalize_source_prefix(body: str) -> str:
    """先頭行のソースタグを正規化する（後方互換）。"""
    import warnings

    lines = body.splitlines()
    if not lines:
        return body
    if lines[0].strip() == "[file-chat]":
        warnings.warn(
            "normalize_source_prefix() は非推奨です。"
            " ソースタグ [file-chat] は廃止済みです。呼び出し元で [file] を使用してください。"
            " v0.10 で削除予定。",
            DeprecationWarning,
            stacklevel=2,
        )
        lines[0] = "[file]"
        return "\n".join(lines)
    return body


def _tail_utf8_bytes(s: str, max_bytes: int) -> str:
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    cut = b[-max_bytes:]
    return cut.decode("utf-8", errors="replace")


def tail_utf8_bytes(s: str, max_bytes: int) -> str:
    """文字列 `s` の末尾 `max_bytes` バイト分を UTF-8 で切り出して返す（public alias）。"""
    return _tail_utf8_bytes(s, max_bytes)


@contextmanager
def persona_memory_lock(
    config: "MemoryConfig",
    persona_stem: str,
    *,
    timeout_sec: float | None = None,
) -> Iterator[bool]:
    if timeout_sec is None:
        timeout_sec = config.lock_timeout_sec
    lock_path = config.chat_dir / f".lock-memory-{persona_stem}"
    config.chat_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_sec
    acquired = False
    while time.monotonic() < deadline:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL, 0o644)
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.05)
        except OSError:
            time.sleep(0.05)
    try:
        yield acquired
    finally:
        if acquired:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass


def _ensure_jsonl(config: "MemoryConfig", persona_stem: str) -> Path:
    """JSONL ファイルパスを返す。"""
    return memory_file_path(config, persona_stem)


def _scan_tail_for_dedupe_key(path: Path, dedupe_key: str) -> bool:
    """JSONL ファイルの末尾付近に dedupe_key が含まれているか確認。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    lines = text.splitlines()
    tail_lines = lines[-MEMORY_DEDUPE_SCAN_LINES:] if len(lines) > MEMORY_DEDUPE_SCAN_LINES else lines
    for line in tail_lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if data.get("dedupe_key") == dedupe_key:
                return True
        except (json.JSONDecodeError, TypeError):
            pass
    return False


def append_memory_entry(
    config: "MemoryConfig",
    persona_stem: str,
    role: str,
    content: str,
    timestamp: str,
    *,
    source_tag: str,
    dedupe_key: str | None = None,
    under_lock: bool = False,
    layer: str | None = None,
) -> bool:
    """JSONL 形式でエントリを追記。

    layer が指定された場合、エントリに layer フィールドを付与する。
    dedupe_key が非空で末尾付近に同一キーがあれば追記しない（冪等、True）。
    ロック取得失敗（under_lock False 時）は False。
    """
    def _write() -> None:
        _resolve_memory_dir(config).mkdir(parents=True, exist_ok=True)
        mp = memory_file_path(config, persona_stem)
        entry = MemoryEntry(
            timestamp=timestamp,
            role=role,
            content=content.strip(),
            source_tag=source_tag,
            layer=layer,
            dedupe_key=dedupe_key,
        )
        line = serialize_entry(entry) + "\n"
        with mp.open("a", encoding="utf-8") as f:
            f.write(line)

    mp = memory_file_path(config, persona_stem)

    if mp.exists():
        try:
            sz = mp.stat().st_size
        except OSError:
            sz = -1
        if 0 < sz <= MEMORY_CORRUPT_THRESHOLD_BYTES:
            _log.error(
                "append_memory_entry: %s is only %d bytes — likely corrupted, "
                "refusing to append (threshold=%d)",
                mp, sz, MEMORY_CORRUPT_THRESHOLD_BYTES,
            )
            return False

    if dedupe_key and mp.exists() and _scan_tail_for_dedupe_key(mp, dedupe_key):
        return True

    if under_lock:
        _write()
        return True

    with persona_memory_lock(config, persona_stem) as ok:
        if not ok:
            return False
        if dedupe_key and mp.exists() and _scan_tail_for_dedupe_key(mp, dedupe_key):
            return True
        _write()
    return True


def read_memory_preferences(
    config: "MemoryConfig",
    persona_stem: str,
    *,
    max_bytes: int | None = None,
) -> str:
    """JSONL から source_tag="preferences" エントリを抽出して返す。"""
    if max_bytes is None:
        max_bytes = config.preferences_max_bytes
    path = _ensure_jsonl(config, persona_stem)
    if not path.exists():
        return ""
    entries = parse_jsonl(path)
    prefs_entries = [e for e in entries if e.source_tag == "preferences"]
    if not prefs_entries:
        return ""
    text = assemble_entries_text(
        prefs_entries,
        preferences_heading=config.preferences_section_name,
    )
    text = text.strip()
    if not text:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="replace")


def read_memory_tail_text(
    config: "MemoryConfig",
    persona_stem: str,
    *,
    max_bytes: int,
    max_entries: int,
    layers: list[str] | None = None,
) -> str:
    """JSONL ファイル末尾から最大 max_entries エントリを返す。

    layers 指定時は layer がリストに含まれるエントリのみ対象。
    """
    if max_bytes == 0:
        return ""
    path = _ensure_jsonl(config, persona_stem)
    if not path.exists():
        return ""
    entries = parse_jsonl(path)
    if not entries:
        return ""
    if layers is not None:
        entries = [e for e in entries if e.layer in layers]
    selected = entries[-max_entries:] if len(entries) > max_entries else entries
    text = assemble_entries_text(
        selected,
        preferences_heading=config.preferences_section_name,
    )
    if not text.strip():
        return ""
    return _tail_utf8_bytes(text, max_bytes)
