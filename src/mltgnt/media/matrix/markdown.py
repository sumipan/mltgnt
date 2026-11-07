"""Markdown to a small HTML subset for Matrix formatted_body."""

from __future__ import annotations

import html
import re

__all__ = ["to_html"]

_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODE_RE = re.compile(r"`([^`]+)`")
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _inline(text: str) -> str:
    # Escape once up front; the substitutions below work on already-escaped text.
    escaped = html.escape(text, quote=True)

    def link_sub(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        if not re.match(r"^https?://", url, re.IGNORECASE):
            return match.group(0)
        return f'<a href="{url}">{label}</a>'

    out = escaped
    out = _LINK_RE.sub(link_sub, out)
    out = _BOLD_RE.sub(r"<strong>\1</strong>", out)
    out = _CODE_RE.sub(r"<code>\1</code>", out)
    return out


def _parse_table(lines: list[str], start: int) -> tuple[str, int] | None:
    if start + 1 >= len(lines):
        return None
    header = lines[start]
    sep = lines[start + 1]
    if "|" not in header or not re.match(r"^\s*\|?[\s\-:|]+\|?[\s\-:|]*\s*$", sep):
        return None
    rows = [header]
    idx = start + 2
    while idx < len(lines) and "|" in lines[idx]:
        rows.append(lines[idx])
        idx += 1

    def cells(line: str) -> list[str]:
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        return parts

    body_rows: list[str] = []
    headers = cells(rows[0])
    body_rows.append("<tr>" + "".join(f"<th>{_inline(h)}</th>" for h in headers) + "</tr>")
    for row in rows[2:]:
        cs = cells(row)
        body_rows.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cs) + "</tr>")
    table = "<table>" + "".join(body_rows) + "</table>"
    return table, idx


def to_html(text: str) -> str:
    """Convert a Markdown subset to HTML (escape raw HTML first)."""
    work = text
    chunks: list[str] = []
    pos = 0
    for match in _FENCE_RE.finditer(text):
        before = text[pos : match.start()]
        if before:
            chunks.append(("text", before))
        chunks.append(("fence", match.group(1)))
        pos = match.end()
    if pos < len(text):
        chunks.append(("text", text[pos:]))
    if not chunks:
        chunks = [("text", work)]

    out_parts: list[str] = []
    for kind, block in chunks:
        if kind == "fence":
            out_parts.append(f"<pre><code>{html.escape(block, quote=True)}</code></pre>")
            continue
        lines = block.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            table = _parse_table(lines, i)
            if table is not None:
                out_parts.append(table[0])
                i = table[1]
                continue
            heading = _HEADING_RE.match(line)
            if heading:
                level = len(heading.group(1))
                out_parts.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
                i += 1
                continue
            if re.match(r"^[-*]\s+", line) or re.match(r"^\d+\.\s+", line):
                items: list[str] = []
                while i < len(lines) and (re.match(r"^[-*]\s+", lines[i]) or re.match(r"^\d+\.\s+", lines[i])):
                    content = re.sub(r"^[-*]\s+", "", lines[i])
                    content = re.sub(r"^\d+\.\s+", "", content)
                    items.append(f"<li>{_inline(content)}</li>")
                    i += 1
                out_parts.append("<ul>" + "".join(items) + "</ul>")
                continue
            para_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip():
                para_lines.append(lines[i])
                i += 1
            inner = "<br>".join(_inline(pl) for pl in para_lines)
            out_parts.append(f"<p>{inner}</p>")
    return "".join(out_parts)
