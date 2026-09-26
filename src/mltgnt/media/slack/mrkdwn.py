"""Markdown -> Slack mrkdwn conversion."""

from __future__ import annotations

import re
from collections.abc import Callable

__all__ = ["markdown_to_mrkdwn", "normalize_markdown_residual"]

_BULLET = "\u2022 "
_CELL_SEP = " / "
# Table separator row: pipes, dashes, colons, spaces and box-drawing characters.
_TABLE_SEP_RE = re.compile(r"^\|[\s\-:|\u2500-\u257f]+\|\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^(\s*)([-*])\s+(\S.*)$")
_ORDERED_RE = re.compile(r"^(\s*)\d+\.\s+(\S.*)$")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]")


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _table_rows_to_bullets(table_lines: list[str]) -> list[str]:
    out: list[str] = []
    for raw in table_lines:
        row = raw.strip()
        if _TABLE_SEP_RE.match(row.replace(" ", "")):
            continue
        cells = [c.strip() for c in row.strip("|").split("|")]
        cells = [c for c in cells if c]
        if cells:
            out.append(_BULLET + _CELL_SEP.join(cells))
    return out


def _flatten_tables(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _is_table_line(lines[i]):
            start = i
            while i < len(lines) and _is_table_line(lines[i]):
                i += 1
            out.extend(_table_rows_to_bullets(lines[start:i]))
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _default_wikilink(target: str, label: str) -> str:
    return label


def markdown_to_mrkdwn(text: str, *, wikilink: Callable[[str, str], str] | None = None) -> str:
    """Convert standard Markdown to Slack mrkdwn.

    Code spans / blocks are kept as-is (fence language dropped). ``[[target|label]]``
    wiki links are rendered by ``wikilink(target, label)``; the default keeps the label.
    """
    t = (text or "").strip()
    if not t:
        return ""
    render_wikilink = wikilink or _default_wikilink

    code_blocks: list[str] = []

    def stash_code(m: re.Match[str]) -> str:
        code_blocks.append(m.group(0))
        return f"\x00CODEBLOCK{len(code_blocks) - 1}\x00"

    t = re.sub(r"```[\s\S]*?```", stash_code, t)

    inline_codes: list[str] = []

    def stash_inline(m: re.Match[str]) -> str:
        inline_codes.append(m.group(0))
        return f"\x00INLINE{len(inline_codes) - 1}\x00"

    t = re.sub(r"`[^`\n]+`", stash_inline, t)
    t = _flatten_tables(t)

    def replace_wikilink(m: re.Match[str]) -> str:
        target = m.group(1).strip()
        label = (m.group(2) or target).strip()
        return render_wikilink(target, label)

    t = _WIKILINK_RE.sub(replace_wikilink, t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", t)

    prev = None
    while prev != t:
        prev = t
        t = re.sub(r"\*\*(.+?)\*\*", r"*\1*", t)
    t = re.sub(r"(?<![\w])__([^_\n]+?)__(?![\w])", r"_\1_", t)
    t = re.sub(r"~~(.+?)~~", r"~\1~", t)

    fixed: list[str] = []
    for line in t.splitlines():
        heading = _HEADING_RE.match(line.strip())
        if heading and heading.group(2).strip():
            fixed.append(f"*{heading.group(2).strip()}*")
            continue
        item = _LIST_RE.match(line)
        if item:
            fixed.append(f"{item.group(1)}{_BULLET}{item.group(3)}")
            continue
        fixed.append(line)
    t = "\n".join(fixed)

    for idx, block in enumerate(code_blocks):
        t = t.replace(f"\x00CODEBLOCK{idx}\x00", re.sub(r"^```\w*", "```", block))
    for idx, code in enumerate(inline_codes):
        t = t.replace(f"\x00INLINE{idx}\x00", code)
    return t.strip()


def normalize_markdown_residual(text: str) -> str:
    """Fallback: turn Markdown that Slack cannot render into near-plain text."""
    t = (text or "").strip()
    if not t:
        return ""
    t = _flatten_tables(t)

    prev = None
    while prev != t:
        prev = t
        t = re.sub(r"\*\*([\s\S]*?)\*\*", r"\1", t)
    t = re.sub(r"(?<!\*)\*(?![\s*])([^*\n]+?)(?<!\*)\*(?!\*)", r"\1", t)
    t = re.sub(r"(?<![\w])_([^_\n]+)_(?![\w])", r"\1", t)

    fixed: list[str] = []
    for line in t.splitlines():
        heading = _HEADING_RE.match(line.strip())
        if heading and heading.group(2).strip():
            fixed.append(heading.group(2).strip())
            continue
        item = _LIST_RE.match(line) or _ORDERED_RE.match(line)
        if item:
            fixed.append(f"{item.group(1)}{_BULLET}{item.groups()[-1]}")
            continue
        fixed.append(line)
    return "\n".join(fixed).strip()
