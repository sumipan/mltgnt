"""
mltgnt.memory — 人物像メモリのパス解決・追記・末尾読込・ロック。

元コード: tools/secretary/memory.py のコアロジック
パス解決は MemoryConfig 引数で受け取る（secretary 固有パスを直接使わない）。

設計: Issue #118 §3 (T3)
"""
from __future__ import annotations

import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from mltgnt.config import MemoryConfig
    from mltgnt.memory._embedding import EmbeddingCall

__all__ = [
    "persona_memory_lock",
    "append_memory_entry",
    "read_memory_preferences",
    "read_memory_tail_text",
    "read_memory_by_relevance",
    "memory_file_path",
    "normalize_source_prefix",
    "compact",
    "needs_compaction",
    "LlmCallError",
    "CompactionResult",
]

_log = logging.getLogger(__name__)

# dedupe 検索: ファイル末尾付近
MEMORY_DEDUPE_SCAN_BYTES = 32 * 1024
MEMORY_DEDUPE_SCAN_LINES = 200

# サイズガード: 既存ファイルがこの閾値以下なら破損とみなし追記を拒否する
# 正常なエントリ1件は最低でも ~40B 以上。10B 以下は改行のみ等の破損状態
MEMORY_CORRUPT_THRESHOLD_BYTES = 10


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

    # サイズガード: 既存ファイルが閾値以下なら破損とみなし追記を拒否する
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


def read_memory_by_relevance(
    config: "MemoryConfig",
    persona_stem: str,
    query: str,
    *,
    max_bytes: int,
    max_entries: int,
    embedding_call: "EmbeddingCall | None" = None,
) -> str:
    """クエリとの関連度が高い memory エントリを選択して返す。

    1. memory ファイルからエントリを `---` で分割
    2. preferences セクションは常に含める（スコアリング対象外）
    3. 残りのエントリを embedding スコアでランク付け
    4. 上位 max_entries 件を max_bytes 以内で結合して返す

    embedding_call が None の場合や API エラー時は
    read_memory_tail_text() にフォールバックする。

    Args:
        config: MemoryConfig
        persona_stem: ペルソナ名
        query: ユーザーの入力テキスト
        max_bytes: 最大バイト数
        max_entries: 最大エントリ数
        embedding_call: embedding API（テスト時に差し替え可能）

    Returns:
        選択された memory テキスト
    """
    # 空クエリまたは embedding_call なしはフォールバック
    if not query or embedding_call is None:
        return read_memory_tail_text(
            config, persona_stem, max_bytes=max_bytes, max_entries=max_entries
        )

    path = memory_file_path(config, persona_stem)
    if not path.exists():
        return ""

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ""

    if not raw.strip():
        return ""

    # `---` で分割してブロックリストを生成
    parts = re.split(r"\n---\s*\n", raw)
    blocks = [p.strip() for p in parts if p.strip()]

    if not blocks:
        return raw.strip()

    # preferences セクションを分離
    prefs_header = "## ユーザーの好み・傾向"
    preferences_blocks: list[str] = []
    entry_blocks: list[str] = []
    for block in blocks:
        if prefs_header in block:
            preferences_blocks.append(block)
        else:
            entry_blocks.append(block)

    # エントリがない場合は preferences のみ結合して返す
    if not entry_blocks:
        result = "\n\n---\n\n".join(preferences_blocks)
        return _tail_utf8_bytes(result, max_bytes)

    # embedding でスコアリング
    try:
        from mltgnt.memory._embedding import get_embeddings
        from mltgnt.memory._scoring import score_entries

        all_texts = [query] + entry_blocks
        all_embeddings = get_embeddings(all_texts, embedding_call=embedding_call)
        query_embedding = all_embeddings[0]
        entry_embeddings = all_embeddings[1:]

        scored = score_entries(query_embedding, entry_embeddings, entry_blocks)
        top_entries = [s.text for s in scored[:max_entries]]

    except Exception as e:
        _log.warning(
            "read_memory_by_relevance: embedding error, fallback to tail text: %s", e
        )
        return read_memory_tail_text(
            config, persona_stem, max_bytes=max_bytes, max_entries=max_entries
        )

    # preferences + 上位エントリを結合
    all_parts = preferences_blocks + top_entries
    text = "\n\n---\n\n".join(all_parts)
    return _tail_utf8_bytes(text, max_bytes)


# Lazy imports for compaction to avoid circular dependency issues
from mltgnt.memory._compaction import (  # noqa: E402
    compact,
    needs_compaction,
    LlmCallError,
    CompactionResult,
)
