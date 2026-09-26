"""Pure helpers that strip internal markers from an agent result before posting.

Every pattern is supplied by the caller; the defaults remove nothing host specific.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable

__all__ = [
    "dedupe_trailing_repeated_block",
    "extract_final_assistant_text",
    "sanitize_result_body",
    "strip_leading_paragraphs",
    "strip_status_lines",
]

_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")


def _assistant_text(obj: object) -> str | None:
    """Concatenated text blocks of a final (non-streaming) assistant event, else None."""
    if not isinstance(obj, dict) or obj.get("type") != "assistant":
        return None
    if "timestamp_ms" in obj or "model_call_id" in obj:
        return None
    message = obj.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None
    texts = [
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and block["text"]
    ]
    return "".join(texts) if texts else None


def extract_final_assistant_text(events_jsonl: str) -> str | None:
    """Return the last aggregated assistant text after the last tool call in an events JSONL.

    Candidates are ``type=assistant`` lines with a non-empty text block and neither
    ``timestamp_ms`` nor ``model_call_id`` (those are streaming deltas).
    """
    if not events_jsonl or not events_jsonl.strip():
        return None
    last_tool_call_idx = -1
    candidates: list[tuple[int, str]] = []
    for idx, line in enumerate(events_jsonl.splitlines()):
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "tool_call":
            last_tool_call_idx = idx
            continue
        text = _assistant_text(obj)
        if text is not None:
            candidates.append((idx, text))
    eligible = [text for i, text in candidates if i > last_tool_call_idx]
    return eligible[-1] if eligible else None


def strip_status_lines(text: str, *, prefixes: Iterable[str] = (), exact: Iterable[str] = ()) -> str:
    """Drop lines whose stripped value equals one of ``exact`` or starts with one of ``prefixes``."""
    prefix_tuple = tuple(prefixes)
    exact_set = frozenset(exact)
    kept = [
        line
        for line in text.splitlines()
        if not (line.strip() in exact_set or (prefix_tuple and line.strip().startswith(prefix_tuple)))
    ]
    return "\n".join(kept).strip()


def strip_leading_paragraphs(body: str, is_noise: Callable[[str], bool]) -> str:
    """Drop blank and ``is_noise`` paragraphs from the start of ``body``."""
    paragraphs = _PARAGRAPH_SPLIT_RE.split(body.strip())
    while paragraphs and (not paragraphs[0].strip() or is_noise(paragraphs[0])):
        paragraphs.pop(0)
    return "\n\n".join(paragraphs).strip()


def dedupe_trailing_repeated_block(body: str, min_repeat_chars: int = 80) -> str:
    """If the last two paragraphs are identical (and long), keep only one."""
    s = body.strip()
    if len(s) < min_repeat_chars * 2 + 4:
        return s
    parts = _PARAGRAPH_SPLIT_RE.split(s)
    if len(parts) < 2:
        return s
    last, prev = parts[-1].strip(), parts[-2].strip()
    if len(last) >= min_repeat_chars and last == prev:
        return "\n\n".join(parts[:-1]).strip()
    return s


def sanitize_result_body(
    text: str,
    *,
    cut_markers: Iterable[str] = (),
    status_prefixes: Iterable[str] = (),
    status_exact: Iterable[str] = (),
    leading_noise: Callable[[str], bool] | None = None,
    formatter: Callable[[str], str] | None = None,
) -> str:
    """Return only the user-facing part of a result body.

    Order: drop leading noise paragraphs, apply ``formatter``, cut at the earliest
    ``cut_markers`` hit, dedupe a repeated trailing block, drop status lines.
    """
    s = text.strip()
    if not s:
        return ""
    if leading_noise is not None:
        s = strip_leading_paragraphs(s, leading_noise)
    if formatter is not None:
        s = formatter(s)
    positions = [p for p in (s.find(m) for m in cut_markers if m) if p != -1]
    if positions:
        s = s[: min(positions)].rstrip()
    s = dedupe_trailing_repeated_block(s)
    return strip_status_lines(s, prefixes=status_prefixes, exact=status_exact)
