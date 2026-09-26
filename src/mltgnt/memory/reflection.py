"""mltgnt.memory.reflection — reflection prompt, parser and applier.

No LLM call lives here: the caller builds the prompt with
``build_reflection_prompt``, runs it through its own ``llm_call(prompt) -> str``,
then feeds the text to ``parse_reflection`` and ``apply_reflection``.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from mltgnt.memory._format import MemoryEntry
from mltgnt.memory.semantic import KINDS, SemanticEntry, SemanticStore, validate_entry

__all__ = [
    "ApplyReport",
    "ReflectionAdd",
    "ReflectionParseError",
    "ReflectionResult",
    "apply_reflection",
    "build_reflection_prompt",
    "parse_reflection",
]

_log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """You are {persona}. Reflect on your recent conversations and update your long-term memory.

## Recent episodes
{episodes}

## Current active memories
{active}

## Task
1. Write a short reflection (1-3 sentences) on what you learned.
2. List new durable memories worth keeping. Skip anything already covered above.
3. List ids of active memories that are now wrong or outdated.

Allowed kinds: {kinds}
Allowed subjects: user | self | unresolved | person:<name> | project:<name> | skill:<name>

## Output
Reply with a single JSON object and nothing else:
{{"reflection": "...", "add": [{{"kind": "...", "content": "...", "subject": "..."}}], "supersede": ["m_..."]}}
"""


class ReflectionParseError(ValueError):
    """The reflection output is not the expected JSON object."""


@dataclass(frozen=True)
class ReflectionAdd:
    kind: str
    content: str
    subject: str


@dataclass(frozen=True)
class ReflectionResult:
    reflection: str
    add: tuple[ReflectionAdd, ...] = ()
    supersede: tuple[str, ...] = ()


@dataclass
class ApplyReport:
    added: list[str] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)
    reflection_id: str | None = None
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (content, reason)
    ignored_supersede: list[str] = field(default_factory=list)
    truncated: int = 0


def _episode_line(entry: MemoryEntry) -> str:
    return f"- {entry.timestamp} {entry.role}: {' '.join(entry.content.split())}"


def _active_line(entry: SemanticEntry) -> str:
    return f"- {entry.id} [{entry.kind}] ({entry.subject}) {' '.join(entry.content.split())}"


def build_reflection_prompt(
    episodes: Sequence[MemoryEntry],
    active: Sequence[SemanticEntry],
    *,
    persona: str,
) -> str:
    return _PROMPT_TEMPLATE.format(
        persona=persona,
        episodes="\n".join(_episode_line(e) for e in episodes) or "(none)",
        active="\n".join(_active_line(e) for e in active) or "(none)",
        kinds=", ".join(KINDS),
    )


def _extract_object(text: str) -> Any:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ReflectionParseError("no JSON object in reflection output")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ReflectionParseError(f"invalid JSON: {exc}") from exc


def parse_reflection(text: str) -> ReflectionResult:
    """Parse the LLM output. Raises ``ReflectionParseError`` on malformed input."""
    data = _extract_object(text)
    if not isinstance(data, dict):
        raise ReflectionParseError("reflection output must be a JSON object")
    reflection = data.get("reflection") or ""
    raw_add = data.get("add") or []
    raw_supersede = data.get("supersede") or []
    if not isinstance(reflection, str):
        raise ReflectionParseError("'reflection' must be a string")
    if not isinstance(raw_add, list) or not all(isinstance(a, dict) for a in raw_add):
        raise ReflectionParseError("'add' must be a list of objects")
    if not isinstance(raw_supersede, list) or not all(isinstance(s, str) for s in raw_supersede):
        raise ReflectionParseError("'supersede' must be a list of strings")
    add = tuple(
        ReflectionAdd(
            kind=str(a.get("kind") or ""),
            content=str(a.get("content") or "").strip(),
            subject=str(a.get("subject") or ""),
        )
        for a in raw_add
    )
    return ReflectionResult(reflection=reflection.strip(), add=add, supersede=tuple(raw_supersede))


def apply_reflection(
    store: SemanticStore,
    result: ReflectionResult,
    *,
    run_id: str,
    ts: str,
    max_add: int = 5,
) -> ApplyReport:
    """Apply supersedes, then adds (at most ``max_add``), then the reflection itself."""
    report = ApplyReport()
    source = f"reflection:{run_id}"
    active_ids = {e.id for e in store.active()}
    for entry_id in result.supersede:
        if entry_id in active_ids and store.supersede(entry_id):
            report.superseded.append(entry_id)
            active_ids.discard(entry_id)
        else:
            _log.warning("apply_reflection: ignoring supersede of missing/inactive id %s", entry_id)
            report.ignored_supersede.append(entry_id)

    adds = result.add
    if len(adds) > max_add:
        report.truncated = len(adds) - max_add
        _log.warning("apply_reflection: truncating %d adds to %d", len(adds), max_add)
        adds = adds[:max_add]
    for add in adds:
        probe = SemanticEntry(id="", ts=ts, kind=add.kind, content=add.content, subject=add.subject, source=source)
        errors = validate_entry(probe)
        if errors:
            report.rejected.append((add.content, errors[0]))
            continue
        if store.find_duplicate(add.content, add.kind, add.subject) is not None:
            report.rejected.append((add.content, "duplicate"))
            continue
        report.added.append(store.append(add.kind, add.content, add.subject, source, ts=ts).id)

    if result.reflection and store.find_duplicate(result.reflection, "reflection", "self") is None:
        report.reflection_id = store.append("reflection", result.reflection, "self", source, ts=ts).id
    return report
