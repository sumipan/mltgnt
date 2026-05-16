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
    "migrate_markdown_to_jsonl",
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


def migrate_markdown_to_jsonl(md_path: Path, jsonl_path: Path) -> int:
    """既存 Markdown メモリファイルを JSONL に変換。変換エントリ数を返す。"""
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError:
        return 0

    entries: list[MemoryEntry] = []

    # section-based 形式を解析
    sections = _parse_memory(text)

    base_ts = "1970-01-01T00:00:00+00:00"

    if sections.preferences:
        body = re.sub(r"^##\s+[^\n]*\n*", "", sections.preferences, count=1).strip()
        if body:
            entries.append(MemoryEntry(
                timestamp=base_ts,
                role="system",
                content=body,
                source_tag="preferences",
            ))

    if sections.long_term:
        body = re.sub(r"^##\s+[^\n]*\n*", "", sections.long_term, count=1).strip()
        if body:
            entries.append(MemoryEntry(
                timestamp=base_ts,
                role="assistant",
                content=body,
                source_tag="compaction",
            ))

    if sections.mid_term:
        body = re.sub(r"^##\s+[^\n]*\n*", "", sections.mid_term, count=1).strip()
        if body:
            entries.append(MemoryEntry(
                timestamp=base_ts,
                role="assistant",
                content=body,
                source_tag="compaction",
            ))

    _DEDUPE_PAT = re.compile(r"<!--\s*memory-dedupe:([^\s>]+)\s*-->")

    def _extract_dedupe_key(body: str) -> tuple[str, str | None]:
        """body から <!-- memory-dedupe:{key} --> を抽出し (cleaned_body, key) を返す。"""
        m = _DEDUPE_PAT.search(body)
        if m:
            key = m.group(1)
            cleaned = _DEDUPE_PAT.sub("", body).strip()
            return cleaned, key
        return body, None

    # recent セクション内の個別エントリをパース
    if sections.recent:
        recent_text = re.sub(r"^##\s+[^\n]*\n*", "", sections.recent, count=1).strip()
        entry_blocks = re.split(r"\n---\s*\n", recent_text)
        entry_pat = re.compile(r"^## (.+?) — (.+?)$", re.MULTILINE)
        for block in entry_blocks:
            block = block.strip()
            if not block:
                continue
            m = entry_pat.match(block)
            if m:
                ts = m.group(1).strip()
                role = m.group(2).strip()
                rest = block[m.end():].strip()
                content_lines = rest.split("\n", 1)
                source_tag = "file"
                body = rest
                if content_lines and re.match(r"^\[.+?\]$", content_lines[0].strip()):
                    source_tag = content_lines[0].strip()[1:-1]
                    body = content_lines[1].strip() if len(content_lines) > 1 else ""
                body, dedupe_key = _extract_dedupe_key(body)
                entries.append(MemoryEntry(
                    timestamp=ts,
                    role=role,
                    content=body,
                    source_tag=source_tag,
                    dedupe_key=dedupe_key,
                ))
            elif block:
                body, dedupe_key = _extract_dedupe_key(block)
                entries.append(MemoryEntry(
                    timestamp=base_ts,
                    role="user",
                    content=body,
                    source_tag="file",
                    dedupe_key=dedupe_key,
                ))

    # エントリ単位形式（section header が一切ない場合）
    if not any([sections.preferences, sections.long_term, sections.mid_term, sections.recent]):
        # 全体を --- で分割してエントリ単位でパース
        entry_pat = re.compile(r"^## (.+?) — (.+?)$", re.MULTILINE)
        entry_blocks = re.split(r"\n---\s*\n", text)
        for block in entry_blocks:
            block = block.strip()
            if not block:
                continue
            m = entry_pat.match(block)
            if m:
                ts = m.group(1).strip()
                role = m.group(2).strip()
                rest = block[m.end():].strip()
                content_lines = rest.split("\n", 1)
                source_tag = "file"
                body = rest
                if content_lines and re.match(r"^\[.+?\]$", content_lines[0].strip()):
                    source_tag = content_lines[0].strip()[1:-1]
                    body = content_lines[1].strip() if len(content_lines) > 1 else ""
                body, dedupe_key = _extract_dedupe_key(body)
                entries.append(MemoryEntry(
                    timestamp=ts,
                    role=role,
                    content=body,
                    source_tag=source_tag,
                    dedupe_key=dedupe_key,
                ))

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(serialize_entry(entry) + "\n")

    return len(entries)


# ---------------------------------------------------------------------------
# Internal helpers (used by migrate_markdown_to_jsonl)
# ---------------------------------------------------------------------------


@dataclass
class _MemorySections:
    preferences: str
    long_term: str
    mid_term: str
    recent: str
    preamble: str = ""


def _parse_memory(
    text: str,
    *,
    preferences_heading: str = _PREFS_HEADING,
) -> _MemorySections:
    """メモリファイルのテキストを 4 層に分解する。migrate_markdown_to_jsonl 専用。"""
    prefs_pat = re.compile(rf"^## {re.escape(preferences_heading)}", re.MULTILINE)
    long_pat = re.compile(r"^## 長期要約", re.MULTILINE)
    mid_pat = re.compile(r"^## 中期要約", re.MULTILINE)
    recent_pat = re.compile(r"^## 直近ログ", re.MULTILINE)

    sections_found: list[tuple[int, str]] = []
    for pat, key in [
        (prefs_pat, "preferences"),
        (long_pat, "long_term"),
        (mid_pat, "mid_term"),
        (recent_pat, "recent"),
    ]:
        m = pat.search(text)
        if m:
            sections_found.append((m.start(), key))

    sections_found.sort(key=lambda x: x[0])

    preamble = ""
    if sections_found:
        preamble = text[: sections_found[0][0]].strip()
    else:
        return _MemorySections(
            preferences="",
            long_term="",
            mid_term="",
            recent="",
            preamble=text.strip(),
        )

    result: dict[str, str] = {"preferences": "", "long_term": "", "mid_term": "", "recent": ""}

    for i, (start, key) in enumerate(sections_found):
        end = sections_found[i + 1][0] if i + 1 < len(sections_found) else len(text)
        section_text = text[start:end]
        section_text = re.sub(r"\n---\s*$", "", section_text.rstrip())
        result[key] = section_text.strip()

    return _MemorySections(
        preferences=result["preferences"],
        long_term=result["long_term"],
        mid_term=result["mid_term"],
        recent=result["recent"],
        preamble=preamble,
    )
