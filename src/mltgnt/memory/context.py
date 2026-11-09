"""mltgnt.memory.context — deterministic memory context assembly."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from mltgnt.config.language import LanguagePack
from mltgnt.memory._format import assemble_entries_text, parse_jsonl
from mltgnt.memory.core_render import render_core
from mltgnt.memory.semantic import SemanticStore

__all__ = ["build_memory_context", "recent_episodes_text", "thread_tail_text"]

_TIMESTAMP_FMT = "%Y-%m-%d %H:%M"

_DEFAULT_CORE_HEADING = "## Memory (core)"
_DEFAULT_THREAD_TAIL_HEADING = "## Recent turns in this thread"
_DEFAULT_RECENT_HEADING = "## Recent conversations (other threads)"


def _thread_needle(*, space_id: str | None, thread_key: str) -> str:
    if space_id is None:
        return f":{thread_key}:"
    return f":{space_id}:{thread_key}:"


def thread_tail_text(
    episodes_path: Path,
    *,
    space_id: str | None,
    thread_key: str,
    max_entries: int,
) -> str:
    """Return the tail of episodic lines for one thread.

    Matching uses a substring of ``dedupe_key`` that aligns with
    ``mltgnt.persona.memory.direct_dedupe_base()`` as
    ``<prefix>:<space_id>:<thread_key>:<message_id>`` (``space_id`` omitted
    when None: ``:<thread_key>:`` only).
    """
    if max_entries <= 0 or not thread_key or not episodes_path.exists():
        return ""
    needle = _thread_needle(space_id=space_id, thread_key=thread_key)
    entries = [e for e in parse_jsonl(episodes_path) if needle in (e.dedupe_key or "")]
    if not entries:
        return ""
    return assemble_entries_text(entries[-max_entries:]).strip()


def _parse_timestamp(ts: str) -> datetime | None:
    try:
        return datetime.strptime(ts, _TIMESTAMP_FMT)
    except ValueError:
        return None


def recent_episodes_text(
    episodes_path: Path,
    *,
    since: datetime,
    max_entries: int,
    exclude_thread_key: str | None = None,
) -> str:
    """Return recent episodic lines since ``since``, optionally excluding one thread."""
    if max_entries <= 0 or not episodes_path.exists():
        return ""
    exclude_needle = f":{exclude_thread_key}:" if exclude_thread_key else None
    selected = []
    for entry in parse_jsonl(episodes_path):
        if exclude_needle and exclude_needle in (entry.dedupe_key or ""):
            continue
        parsed = _parse_timestamp(entry.timestamp)
        if parsed is None or parsed < since:
            continue
        selected.append(entry)
    if not selected:
        return ""
    return assemble_entries_text(selected[-max_entries:]).strip()


def _resolve_heading(
    explicit: str | None,
    pack: LanguagePack | None,
    pack_attr: str,
    default: str,
) -> str:
    if explicit is not None:
        return explicit
    if pack is not None:
        return getattr(pack, pack_attr)
    return default


def build_memory_context(
    persona_dir: Path,
    *,
    persona_stem: str,
    space_id: str | None,
    thread_key: str | None,
    core_max_bytes: int = 4096,
    thread_tail_entries: int = 5,
    recent_hours: int = 0,
    recent_entries: int = 0,
    now: datetime | None = None,
    pack: LanguagePack | None = None,
    core_heading: str | None = None,
    thread_tail_heading: str | None = None,
    recent_heading: str | None = None,
) -> str:
    """Assemble semantic core and optional episodic sections (no LLM)."""
    parts: list[str] = []

    core_heading_resolved = _resolve_heading(
        core_heading, pack, "memory_core_heading", _DEFAULT_CORE_HEADING,
    )
    thread_heading_resolved = _resolve_heading(
        thread_tail_heading, pack, "memory_thread_tail_heading", _DEFAULT_THREAD_TAIL_HEADING,
    )
    recent_heading_resolved = _resolve_heading(
        recent_heading, pack, "memory_recent_heading", _DEFAULT_RECENT_HEADING,
    )

    store = SemanticStore(persona_dir / "semantic.jsonl", persona_stem=persona_stem)
    core = render_core(store.active(), max_bytes=core_max_bytes, heading=core_heading_resolved)
    if core:
        parts.append(core)

    episodes_path = persona_dir / "episodes.jsonl"
    if thread_key and thread_tail_entries > 0:
        tail = thread_tail_text(
            episodes_path,
            space_id=space_id,
            thread_key=thread_key,
            max_entries=thread_tail_entries,
        )
        if tail:
            parts.append(f"{thread_heading_resolved}\n{tail}")

    if recent_hours > 0 and recent_entries > 0:
        anchor = now or datetime.now()
        recent = recent_episodes_text(
            episodes_path,
            since=anchor - timedelta(hours=recent_hours),
            max_entries=recent_entries,
            exclude_thread_key=thread_key,
        )
        if recent:
            parts.append(f"{recent_heading_resolved}\n{recent}")

    if not parts:
        return ""
    return "\n\n".join(parts)
