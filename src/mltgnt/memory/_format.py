"""
mltgnt.memory._format — メモリファイルのパース・フォーマット。

設計: Issue #823 (JSONL 統一)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "MemoryEntry",
    "parse_jsonl",
    "serialize_entry",
    "assemble_entries_text",
]


@dataclass
class MemoryEntry:
    timestamp: str
    role: str
    content: str
    source_tag: str
    layer: str | None = None
    dedupe_key: str | None = None


def serialize_entry(entry: MemoryEntry) -> str:
    """MemoryEntry を JSON 1行に変換。null フィールドは省略。"""
    d: dict[str, Any] = {
        "timestamp": entry.timestamp,
        "role": entry.role,
        "content": entry.content,
        "source_tag": entry.source_tag,
    }
    if entry.layer is not None:
        d["layer"] = entry.layer
    if entry.dedupe_key is not None:
        d["dedupe_key"] = entry.dedupe_key
    return json.dumps(d, ensure_ascii=False)


def parse_jsonl(path: Path) -> list[MemoryEntry]:
    """JSONL ファイルを MemoryEntry リストに変換。不正行はスキップ。"""
    entries: list[MemoryEntry] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return entries
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            entries.append(MemoryEntry(
                timestamp=data.get("timestamp", ""),
                role=data.get("role", ""),
                content=data.get("content", ""),
                source_tag=data.get("source_tag", ""),
                layer=data.get("layer"),
                dedupe_key=data.get("dedupe_key"),
            ))
        except (json.JSONDecodeError, TypeError):
            pass
    return entries


_PREFS_HEADING = "ユーザーの好み・傾向"


def assemble_entries_text(
    entries: list[MemoryEntry],
    *,
    preferences_heading: str = _PREFS_HEADING,
) -> str:
    """MemoryEntry リストを表示用テキストに変換。

    source_tag="preferences" のエントリは `## {preferences_heading}` 見出しで出力し、
    他のエントリは `## {timestamp} — {role}` 形式で出力する。
    エントリは `---` で区切られる。
    """
    parts: list[str] = []
    for entry in entries:
        if entry.source_tag == "preferences":
            parts.append(f"## {preferences_heading}\n\n{entry.content.strip()}")
        else:
            body = f"[{entry.source_tag}]\n{entry.content.strip()}" if entry.content.strip() else f"[{entry.source_tag}]"
            parts.append(f"## {entry.timestamp} — {entry.role}\n\n{body}")
    if not parts:
        return ""
    return "\n\n---\n\n".join(parts) + "\n"


