"""mltgnt.memory.tools — ``remember`` / ``recall`` / ``forget`` tool executor.

``MemoryToolExecutor`` has the ``mltgnt.agent._runner.ToolExecutor`` shape
(``(tool_name, tool_args) -> str``) and never raises: every outcome, including
rejections and internal errors, is returned as a JSON or text string.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mltgnt.memory._format import parse_jsonl
from mltgnt.memory.semantic import KINDS, SemanticEntry, SemanticStore, validate_entry

__all__ = [
    "MEMORY_TOOL_NAMES",
    "MEMORY_TOOL_SPECS",
    "MemoryGate",
    "MemoryToolExecutor",
    "query_terms",
]

_log = logging.getLogger(__name__)

RECALL_DEFAULT_LIMIT = 10
RECALL_MAX_LIMIT = 20

_SUBJECT_DESCRIPTION = (
    "Who the memory is about: user | self | unresolved | person:<name> | project:<name> | skill:<name>"
)

MEMORY_TOOL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "remember",
        "description": (
            "Store one durable memory (a fact, preference, commitment, caveat, "
            "self-observation or reflection) for later conversations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory, one short sentence."},
                "kind": {"type": "string", "enum": list(KINDS)},
                "subject": {"type": "string", "description": _SUBJECT_DESCRIPTION},
            },
            "required": ["content", "kind", "subject"],
            "additionalProperties": False,
        },
    },
    {
        "name": "recall",
        "description": "Search stored memories and past conversation episodes.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Words to search for."},
                "kind": {"type": "string", "enum": list(KINDS)},
                "subject": {"type": "string", "description": _SUBJECT_DESCRIPTION},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": RECALL_MAX_LIMIT,
                    "default": RECALL_DEFAULT_LIMIT,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "forget",
        "description": (
            "Retire one stored memory by id, or by text contained in exactly one active memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Memory id such as m_2026-09-26_0001."},
                "text": {"type": "string", "description": "Text contained in the memory."},
            },
            "additionalProperties": False,
        },
    },
)

MEMORY_TOOL_NAMES: frozenset[str] = frozenset(spec["name"] for spec in MEMORY_TOOL_SPECS)


@dataclass(frozen=True)
class MemoryGate:
    """Limits applied to ``remember``.

    ``secret_check`` returns True when the content looks like a secret.
    ``allowed_subject_prefixes=None`` allows every valid subject.
    """

    max_per_turn: int = 3
    max_chars: int = 300
    secret_check: Callable[[str], bool] | None = None
    allowed_subject_prefixes: tuple[str, ...] | None = None


def query_terms(query: str) -> list[str]:
    """Whitespace-split terms; non-ASCII terms become character 2-grams."""
    terms: list[str] = []
    for token in query.casefold().split():
        if token.isascii() or len(token) < 2:
            terms.append(token)
        else:
            terms.extend(token[i:i + 2] for i in range(len(token) - 1))
    return list(dict.fromkeys(terms))


def _score(terms: list[str], text: str) -> int:
    folded = text.casefold()
    return sum(1 for term in terms if term in folded)


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


class MemoryToolExecutor:
    """Executes the three memory tools against a ``SemanticStore``.

    ``source`` is either a fixed string or a callable returning the current
    source (e.g. ``slack:<ch>:<ts>``). ``subject_resolver`` maps the subject
    given by the model to a canonical one before validation.
    """

    def __init__(
        self,
        store: SemanticStore,
        *,
        episodes_path: Path,
        gate: MemoryGate,
        source: str | Callable[[], str],
        now: Callable[[], datetime],
        subject_resolver: Callable[[str], str] | None = None,
    ) -> None:
        self.store = store
        self.episodes_path = Path(episodes_path)
        self.gate = gate
        self._source = source
        self._now = now
        self._subject_resolver = subject_resolver
        self._turn_count = 0

    def reset_turn(self) -> None:
        self._turn_count = 0

    def __call__(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        try:
            args = tool_args if isinstance(tool_args, dict) else {}
            if tool_name == "remember":
                return _dumps(self._remember(args))
            if tool_name == "recall":
                return self._recall(args)
            if tool_name == "forget":
                return _dumps(self._forget(args))
            return _dumps({"ok": False, "reason": "unknown_tool"})
        except Exception as exc:  # noqa: BLE001 — tool executors must not raise
            _log.warning("memory tool %s failed: %s", tool_name, exc)
            return _dumps({"ok": False, "reason": "error", "detail": str(exc)})

    # -- remember ---------------------------------------------------------

    def _reject(self, reason: str, kind: str, subject: str) -> dict[str, Any]:
        _log.warning("memory remember rejected: reason=%s kind=%s subject=%s", reason, kind, subject)
        return {"ok": False, "reason": reason}

    def _remember(self, args: dict[str, Any]) -> dict[str, Any]:
        content = str(args.get("content") or "").strip()
        kind = str(args.get("kind") or "")
        subject = str(args.get("subject") or "")
        if self._subject_resolver is not None and subject:
            subject = self._subject_resolver(subject)
        gate = self.gate
        if self._turn_count >= gate.max_per_turn:
            return self._reject("per_turn_limit", kind, subject)
        if len(content) > gate.max_chars:
            return self._reject("too_long", kind, subject)
        probe = SemanticEntry(id="", ts="", kind=kind, content=content, subject=subject, source="")
        errors = validate_entry(probe)
        for reason in ("invalid_kind", "invalid_subject", "empty_content"):
            if reason in errors:
                return self._reject(reason, kind, subject)
        prefixes = gate.allowed_subject_prefixes
        if prefixes is not None and not subject.startswith(tuple(prefixes)):
            return self._reject("subject_not_allowed", kind, subject)
        if gate.secret_check is not None and gate.secret_check(content):
            return self._reject("secret", kind, subject)
        if self.store.find_duplicate(content, kind, subject) is not None:
            return self._reject("duplicate", kind, subject)
        source = self._source() if callable(self._source) else self._source
        entry = self.store.append(
            kind, content, subject, source, ts=self._now().strftime("%Y-%m-%d %H:%M")
        )
        self._turn_count += 1
        return {"ok": True, "id": entry.id}

    # -- recall -----------------------------------------------------------

    def _recall(self, args: dict[str, Any]) -> str:
        terms = query_terms(str(args.get("query") or ""))
        kind = args.get("kind") or None
        subject = args.get("subject") or None
        if subject is not None and self._subject_resolver is not None:
            subject = self._subject_resolver(str(subject))
        try:
            limit = int(args.get("limit") or RECALL_DEFAULT_LIMIT)
        except (TypeError, ValueError):
            limit = RECALL_DEFAULT_LIMIT
        limit = max(1, min(limit, RECALL_MAX_LIMIT))

        # (score, recency key, line)
        hits: list[tuple[int, str, str]] = []
        for entry in self.store.active():
            if kind is not None and entry.kind != kind:
                continue
            if subject is not None and entry.subject != subject:
                continue
            score = _score(terms, entry.content)
            if terms and score == 0:
                continue
            line = f"[{entry.id}] [{entry.kind}] ({entry.subject}) {entry.content}"
            hits.append((score, entry.ts, line))
        if kind is None and subject is None:
            for episode in parse_jsonl(self.episodes_path):
                score = _score(terms, episode.content)
                if terms and score == 0:
                    continue
                content = " ".join(episode.content.split())
                line = f"[episode {episode.timestamp}] {episode.role}: {content}"
                hits.append((score, episode.timestamp, line))
        if not hits:
            return "No matching memories."
        hits.sort(key=lambda h: (h[0], h[1]), reverse=True)
        return "\n".join(line for _, _, line in hits[:limit])

    # -- forget -----------------------------------------------------------

    def _forget(self, args: dict[str, Any]) -> dict[str, Any]:
        entry_id = str(args.get("id") or "").strip()
        text = str(args.get("text") or "").strip()
        if entry_id:
            if self.store.supersede(entry_id):
                return {"ok": True, "id": entry_id}
            return {"ok": False, "reason": "not_found"}
        if not text:
            return {"ok": False, "reason": "invalid_args"}
        needle = text.casefold()
        matches = [e for e in self.store.active() if needle in e.content.casefold()]
        if not matches:
            return {"ok": False, "reason": "not_found"}
        if len(matches) > 1:
            return {"ok": False, "reason": "ambiguous", "candidates": [e.id for e in matches]}
        self.store.supersede(matches[0].id)
        return {"ok": True, "id": matches[0].id}
