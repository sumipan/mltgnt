"""mltgnt.memory.core_render — render the always-loaded memory core.

caveat and commitment entries are always included (a WARNING is logged when they
alone exceed ``max_bytes``); the remaining budget is filled with preference,
self, fact and reflection entries, newest first within each kind.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

from mltgnt.memory.semantic import SemanticEntry

__all__ = ["MANDATORY_KINDS", "OPTIONAL_KINDS", "render_core"]

_log = logging.getLogger(__name__)

MANDATORY_KINDS: tuple[str, ...] = ("caveat", "commitment")
OPTIONAL_KINDS: tuple[str, ...] = ("preference", "self", "fact", "reflection")


def _line(entry: SemanticEntry) -> str:
    content = " ".join(entry.content.split())
    return f"- [{entry.kind}] {content}"


def _newest_first(entries: list[SemanticEntry], kind: str) -> list[SemanticEntry]:
    return sorted((e for e in entries if e.kind == kind), key=lambda e: (e.ts, e.id), reverse=True)


def _size(lines: list[str]) -> int:
    return len("\n".join(lines).encode("utf-8"))


def render_core(
    entries: Iterable[SemanticEntry],
    *,
    max_bytes: int = 4096,
    heading: str = "## Memory",
) -> str:
    """Render active entries as ``heading`` + ``- [<kind>] <content>`` lines."""
    active = [e for e in entries if e.status == "active"]
    if not active:
        return ""
    lines = [heading]
    for kind in MANDATORY_KINDS:
        lines.extend(_line(e) for e in _newest_first(active, kind))
    size = _size(lines)
    if size > max_bytes:
        _log.warning(
            "render_core: caveat/commitment alone take %d bytes (max_bytes=%d); not truncated",
            size, max_bytes,
        )
    for kind in OPTIONAL_KINDS:
        for entry in _newest_first(active, kind):
            line = _line(entry)
            added = len(line.encode("utf-8")) + 1
            if size + added <= max_bytes:
                lines.append(line)
                size += added
    if len(lines) == 1:
        return ""
    return "\n".join(lines)
