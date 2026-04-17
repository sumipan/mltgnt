"""
mltgnt.memory — 人物像メモリのパス解決・追記・末尾読込・ロック。

元コード: tools/secretary/memory.py のコアロジック
パス解決は MemoryConfig 引数で受け取る（secretary 固有パスを直接使わない）。

設計: Issue #118 §3 (T3)
"""
from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from mltgnt.config import MemoryConfig

__all__ = [
    "persona_memory_lock",
    "append_memory_entry",
    "read_memory_preferences",
    "read_memory_tail_text",
    "memory_file_path",
    "normalize_source_prefix",
    "compact",
    "needs_compaction",
    "LlmCallError",
    "CompactionResult",
]

# dedupe 検索: ファイル末尾付近
MEMORY_DEDUPE_SCAN_BYTES = 32 * 1024
MEMORY_DEDUPE_SCAN_LINES = 200


def _resolve_memory_dir(config: "MemoryConfig") -> Path:
    """config.chat_memory_dir が None の場合は chat_dir / "memory" を返す。"""
    if config.chat_memory_dir is not None:
        return config.chat_memory_dir
    return config.chat_dir / "memory"


def memory_file_path(config: "MemoryConfig", persona_stem: str) -> Path:
    """`_resolve_memory_dir(config) / f\"{persona_stem}.md\"`"""
    return _resolve_memory_dir(config) / f"{persona_stem}.md"


def normalize_source_prefix(body: str) -> str:
    """先頭行のソースタグを正規化する。

    認識済みタグ: [file], [slack], [scheduled]
    後方互換: [file-chat] → [file] に変換。
    """
    lines = body.splitlines()
    if not lines:
        return body
    if lines[0].strip() == "[file-chat]":
        lines[0] = "[file]"
        return "\n".join(lines)
    return body


def _tail_utf8_bytes(s: str, max_bytes: int) -> str:
    """UTF-8 で末尾 max_bytes に収まるように先頭を切り詰める。"""
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    cut = b[-max_bytes:]
    return cut.decode("utf-8", errors="replace")


@contextmanager
def persona_memory_lock(
    config: "MemoryConfig",
    persona_stem: str,
    *,
    timeout_sec: float | None = None,
) -> Iterator[bool]:
    """`config.chat_dir/.lock-memory-{persona_stem}` を排他作成で取得し、finally で削除。

    タイムアウト時はコンテキストに入り `yield False`（取得失敗）。
    """
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


def _scan_tail_for_dedupe_key(path: Path, dedupe_key: str) -> bool:
    """末尾付近に dedupe_key が含まれていれば True（重複あり）。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    lines = text.splitlines()
    tail_lines = lines[-MEMORY_DEDUPE_SCAN_LINES:] if len(lines) > MEMORY_DEDUPE_SCAN_LINES else lines
    tail_text = "\n".join(tail_lines)
    if len(tail_text.encode("utf-8")) > MEMORY_DEDUPE_SCAN_BYTES:
        b = tail_text.encode("utf-8")
        tail_text = b[-MEMORY_DEDUPE_SCAN_BYTES:].decode("utf-8", errors="replace")
    return dedupe_key in tail_text


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
) -> bool:
    """§3.1 形式で追記。本文先頭行は source_tag。

    `under_lock` が False のとき内部で `persona_memory_lock` を取得する。
    True のときは呼び出し側が既にロックを保持していること。

    dedupe_key が非空で末尾付近に同一キーがあれば追記しない（冪等、True）。
    ロック取得失敗（under_lock False 時）は False。
    """
    def _write() -> None:
        _resolve_memory_dir(config).mkdir(parents=True, exist_ok=True)
        mp = memory_file_path(config, persona_stem)
        body = content.strip()
        if dedupe_key:
            marker = f"<!-- memory-dedupe:{dedupe_key} -->"
            inner = (
                f"{source_tag}\n{marker}\n{body}" if body else f"{source_tag}\n{marker}"
            )
        else:
            inner = f"{source_tag}\n{body}" if body else source_tag
        entry = f"## {timestamp} — {role}\n\n{inner}\n\n---\n\n"
        with mp.open("a", encoding="utf-8") as f:
            f.write(entry)

    mp = memory_file_path(config, persona_stem)
    if dedupe_key and mp.exists() and _scan_tail_for_dedupe_key(mp, dedupe_key):
        return True

    if under_lock:
        _write()
        return True

    with persona_memory_lock(config, persona_stem) as ok:
        if not ok:
            return False
        # ロック取得後に再チェック（競合で直前に追記された場合）
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
    """ファイル冒頭の `## ユーザーの好み・傾向` セクションを抽出して返す。

    セクションが存在しない場合は空文字列を返す。
    抽出結果が max_bytes を超える場合は UTF-8 バイト単位で末尾を切り詰める。
    """
    if max_bytes is None:
        max_bytes = config.preferences_max_bytes
    path = memory_file_path(config, persona_stem)
    if not path.exists():
        return ""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ""

    # 好みセクションの開始を探す
    prefs_header = "## ユーザーの好み・傾向"
    start_idx = raw.find(prefs_header)
    if start_idx == -1:
        return ""

    # セクション終端を探す（次の `---` 区切りまで）
    section_text = raw[start_idx:]
    sep_match = re.search(r"\n---\s*\n", section_text)
    if sep_match:
        section_text = section_text[: sep_match.start()]

    section_text = section_text.strip()
    if not section_text:
        return ""

    # max_bytes を超えた場合は末尾を切り詰める
    encoded = section_text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return section_text
    return encoded[:max_bytes].decode("utf-8", errors="replace")


def read_memory_tail_text(
    config: "MemoryConfig",
    persona_stem: str,
    *,
    max_bytes: int,
    max_entries: int,
) -> str:
    """ファイル末尾からエントリ単位で最大 max_entries 個、合計おおよそ max_bytes まで。"""
    if max_bytes == 0:
        return ""
    path = memory_file_path(config, persona_stem)
    if not path.exists():
        return ""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if not raw.strip():
        return ""
    parts = re.split(r"\n---\s*\n", raw)
    blocks = [p.strip() for p in parts if p.strip()]
    if not blocks:
        return raw.strip()[:max_bytes] if max_bytes else ""
    selected = blocks[-max_entries:] if len(blocks) > max_entries else blocks
    text = "\n\n---\n\n".join(selected)
    return _tail_utf8_bytes(text, max_bytes)


# Lazy imports for compaction to avoid circular dependency issues
from mltgnt.memory._compaction import (  # noqa: E402
    compact,
    needs_compaction,
    LlmCallError,
    CompactionResult,
)
