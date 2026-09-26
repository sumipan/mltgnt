"""Text extraction from Slack messages (blocks / attachments / tables) and thread fetch."""

from __future__ import annotations

import logging
from typing import Any

__all__ = [
    "extract_text_from_blocks",
    "extract_text_from_message",
    "fetch_thread_messages",
    "table_block_to_markdown",
]

_log = logging.getLogger(__name__)

_SLACK_API_MAX = 1000  # conversations_replies limit


def _rich_text_display(node: object) -> str:
    """Display text of a rich_text node, recursively."""
    if not isinstance(node, dict):
        return ""
    ntype = node.get("type")
    if ntype == "text":
        return str(node.get("text") or "")
    if ntype == "link":
        return str(node.get("text") or node.get("url") or "")
    elements = node.get("elements")
    if not isinstance(elements, list):
        return ""
    return "".join(_rich_text_display(child) for child in elements)


def _table_cell_text(cell: object) -> str:
    if not isinstance(cell, dict):
        return ""
    ctype = cell.get("type")
    if ctype == "raw_text":
        return str(cell.get("text") or "")
    if ctype == "rich_text":
        return _rich_text_display(cell)
    return ""


def table_block_to_markdown(block: dict[str, Any]) -> str:
    """Convert a Slack ``type=table`` block into a Markdown table (empty rows dropped)."""
    if not isinstance(block, dict) or block.get("type") != "table":
        return ""
    rows = block.get("rows")
    if not isinstance(rows, list):
        return ""
    parsed: list[list[str]] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        cells = [_table_cell_text(cell).replace("|", "\\|").replace("\n", "<br>") for cell in row]
        if any(cells):
            parsed.append(cells)
    if not parsed:
        return ""
    max_cols = max(len(row) for row in parsed)
    for row in parsed:
        row.extend([""] * (max_cols - len(row)))
    lines = [
        "| " + " | ".join(parsed[0]) + " |",
        "| " + " | ".join(["---"] * max_cols) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in parsed[1:])
    return "\n".join(lines)


def extract_text_from_blocks(blocks: object) -> str:
    """Text of rich_text / section / table blocks, one block per line."""
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        text = ""
        if btype == "rich_text":
            text = _rich_text_display(block)
        elif btype == "section":
            text_obj = block.get("text")
            if isinstance(text_obj, dict):
                text = str(text_obj.get("text") or "")
        elif btype == "table":
            text = table_block_to_markdown(block)
        if text.strip():
            parts.append(text.strip())
    return "\n".join(parts)


def _extract_attachment_layer(attachments: object) -> str:
    """Attachment blocks if any, else attachment fallback / text / pretext."""
    if not isinstance(attachments, list):
        return ""
    block_parts: list[str] = []
    fallback_parts: list[str] = []
    for att in attachments:
        if not isinstance(att, dict):
            continue
        blocks_text = extract_text_from_blocks(att.get("blocks"))
        if blocks_text:
            block_parts.append(blocks_text)
            continue
        fallback = att.get("fallback") or att.get("text") or att.get("pretext") or ""
        if isinstance(fallback, str) and fallback.strip():
            fallback_parts.append(fallback.strip())
    return "\n".join(block_parts or fallback_parts)


def extract_text_from_message(message: dict[str, Any]) -> str:
    """Display text of a Slack message / event.

    First non-empty layer wins: ``text``, top-level blocks, attachment blocks,
    attachment fallback / text / pretext.
    """
    if not isinstance(message, dict):
        return ""
    text = message.get("text")
    if isinstance(text, str) and text.strip():
        return text
    blocks_text = extract_text_from_blocks(message.get("blocks"))
    if blocks_text.strip():
        return blocks_text
    return _extract_attachment_layer(message.get("attachments")).strip()


def fetch_thread_messages(
    client: Any,
    channel: str,
    thread_ts: str,
    *,
    limit: int | None = None,
    exclude_ts: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch a thread via ``conversations_replies``.

    Returns ``[{"user", "text", "ts", "is_bot", "bot_id"}, ...]`` (oldest first, the
    latest ``limit`` when given), or ``[]`` on API error.
    """
    try:
        resp = client.conversations_replies(
            channel=channel, ts=thread_ts, limit=limit if limit is not None else _SLACK_API_MAX
        )
    except Exception as exc:
        _log.warning("slack history: conversations_replies failed: %s", exc)
        return []
    result: list[dict[str, Any]] = []
    for m in resp.get("messages", []):
        ts = m.get("ts", "")
        if exclude_ts is not None and ts == exclude_ts:
            continue
        bot_id = m.get("bot_id") or None
        result.append(
            {
                "user": m.get("user", ""),
                "text": extract_text_from_message(m),
                "ts": ts,
                "is_bot": bot_id is not None,
                "bot_id": bot_id,
            }
        )
    if limit is not None and len(result) > limit:
        result = result[-limit:]
    return result
