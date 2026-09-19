"""
mltgnt.memory._format — parse and format memory files.

Design: Issue #823 (JSONL unification)
"""

from __future__ import annotations

import json
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
    """Convert a MemoryEntry to one JSON line. Omit null fields."""
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
    """Convert a JSONL file to a MemoryEntry list. Skip bad lines."""
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
            entries.append(
                MemoryEntry(
                    timestamp=data.get("timestamp", ""),
                    role=data.get("role", ""),
                    content=data.get("content", ""),
                    source_tag=data.get("source_tag", ""),
                    layer=data.get("layer"),
                    dedupe_key=data.get("dedupe_key"),
                )
            )
        except (json.JSONDecodeError, TypeError):
            pass
    return entries


_PREFS_HEADING = "User’s preferences and tendencies"


def assemble_entries_text(
    entries: list[MemoryEntry],
    *,
    preferences_heading: str = _PREFS_HEADING,
) -> str:
    """Convert a MemoryEntry list to display text.

    Entries with source_tag="preferences" use a `## {preferences_heading}` heading;
    others use `## {timestamp} — {role}`.
    Entries are separated by `---`.
    """
    parts: list[str] = []
    for entry in entries:
        if entry.source_tag == "preferences":
            parts.append(f"## {preferences_heading}\n\n{entry.content.strip()}")
        else:
            body = (
                f"[{entry.source_tag}]\n{entry.content.strip()}" if entry.content.strip() else f"[{entry.source_tag}]"
            )
            parts.append(f"## {entry.timestamp} — {entry.role}\n\n{body}")
    if not parts:
        return ""
    return "\n\n---\n\n".join(parts) + "\n"
