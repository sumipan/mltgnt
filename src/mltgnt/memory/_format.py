"""
mltgnt.memory._format — メモリファイルの 4 層パース・フォーマット。

設計: Issue #123
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "MemorySections",
    "parse_memory",
    "format_memory",
]


@dataclass
class MemorySections:
    preferences: str
    long_term: str
    mid_term: str
    recent: str
    preamble: str = ""


def parse_memory(
    text: str,
    *,
    preferences_heading: str = "ユーザーの好み・傾向",
) -> MemorySections:
    """メモリファイルのテキストを 4 層に分解する。"""
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
        return MemorySections(
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
        # Remove trailing --- separator and whitespace
        section_text = re.sub(r"\n---\s*$", "", section_text.rstrip())
        result[key] = section_text.strip()

    return MemorySections(
        preferences=result["preferences"],
        long_term=result["long_term"],
        mid_term=result["mid_term"],
        recent=result["recent"],
        preamble=preamble,
    )


def format_memory(sections: MemorySections) -> str:
    """MemorySections を 1 つのメモリファイルテキストに結合する。"""
    parts: list[str] = []
    if sections.preamble:
        parts.append(sections.preamble)
    for content in [
        sections.preferences,
        sections.long_term,
        sections.mid_term,
        sections.recent,
    ]:
        if content:
            parts.append(content)
    return "\n\n---\n\n".join(parts) + "\n"
