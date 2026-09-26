"""mltgnt.memory.semantic — semantic memory store (``semantic.jsonl``).

One entry per line::

    {"id": "m_2026-09-26_0042", "ts": "2026-09-26 14:05", "kind": "commitment",
     "content": "...", "subject": "user", "source": "slack:<ch>:<ts>",
     "status": "active", "supersedes": "m_..."}

Appends write a single line; ``supersede`` rewrites the file through a tmp file
and ``os.replace``. Nothing is deleted physically.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from mltgnt.memory._commit import schedule_commit

__all__ = [
    "KINDS",
    "STATUSES",
    "SemanticEntry",
    "SemanticStore",
    "normalize_content",
    "validate_entry",
]

_log = logging.getLogger(__name__)

KINDS: tuple[str, ...] = ("fact", "preference", "commitment", "caveat", "self", "reflection")
STATUSES: tuple[str, ...] = ("active", "superseded")

_SUBJECT_RE = re.compile(r"^(?:user|self|unresolved|(?:person|project|skill):\S.*)$")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SemanticEntry:
    id: str
    ts: str
    kind: str
    content: str
    subject: str
    source: str
    status: str = "active"
    supersedes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["supersedes"] is None:
            del d["supersedes"]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticEntry":
        return cls(
            id=str(data.get("id", "")),
            ts=str(data.get("ts", "")),
            kind=str(data.get("kind", "")),
            content=str(data.get("content", "")),
            subject=str(data.get("subject", "")),
            source=str(data.get("source", "")),
            status=str(data.get("status", "active")),
            supersedes=data.get("supersedes"),
        )


def normalize_content(content: str) -> str:
    """Collapse whitespace runs to one space and casefold (duplicate key)."""
    return _WS_RE.sub(" ", content).strip().casefold()


def validate_entry(entry: SemanticEntry) -> list[str]:
    """Return violation codes: ``invalid_kind`` / ``invalid_subject`` / ``empty_content``."""
    errors: list[str] = []
    if entry.kind not in KINDS:
        errors.append("invalid_kind")
    if not _SUBJECT_RE.match(entry.subject):
        errors.append("invalid_subject")
    if not entry.content.strip():
        errors.append("empty_content")
    return errors


class SemanticStore:
    """Append-only JSONL store of semantic memories for one persona.

    ``commit_debounce_sec=None`` disables the debounced git commit of the file.
    """

    def __init__(
        self,
        path: Path,
        *,
        persona_stem: str,
        commit_debounce_sec: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.persona_stem = persona_stem
        self.commit_debounce_sec = commit_debounce_sec

    def entries(self) -> list[SemanticEntry]:
        """All entries in file order. Broken lines are skipped."""
        return [e for _, e in self._read_lines() if e is not None]

    def active(self) -> list[SemanticEntry]:
        return [e for e in self.entries() if e.status == "active"]

    def get(self, entry_id: str) -> SemanticEntry | None:
        for entry in self.entries():
            if entry.id == entry_id:
                return entry
        return None

    def find_duplicate(self, content: str, kind: str, subject: str) -> SemanticEntry | None:
        """Active entry with the same kind, subject and normalized content."""
        key = normalize_content(content)
        for entry in self.active():
            if entry.kind == kind and entry.subject == subject and normalize_content(entry.content) == key:
                return entry
        return None

    def append(
        self,
        kind: str,
        content: str,
        subject: str,
        source: str,
        *,
        ts: str,
        supersedes: str | None = None,
    ) -> SemanticEntry:
        """Validate and append one active entry. Raises ``ValueError`` on violations."""
        entry = SemanticEntry(
            id="",
            ts=ts,
            kind=kind,
            content=content.strip(),
            subject=subject,
            source=source,
            supersedes=supersedes,
        )
        errors = validate_entry(entry)
        if errors:
            raise ValueError(", ".join(errors))
        entry = replace(entry, id=self._next_id(ts[:10]))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        self._schedule_commit("remember")
        return entry

    def supersede(self, entry_id: str) -> bool:
        """Mark an active entry superseded. False when missing or not active."""
        lines = self._read_lines()
        found = False
        out: list[str] = []
        for raw, entry in lines:
            if entry is not None and not found and entry.id == entry_id and entry.status == "active":
                found = True
                raw = json.dumps(replace(entry, status="superseded").to_dict(), ensure_ascii=False)
            out.append(raw)
        if not found:
            return False
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text("".join(line + "\n" for line in out), encoding="utf-8")
        os.replace(tmp, self.path)
        self._schedule_commit("supersede")
        return True

    def _next_id(self, date: str) -> str:
        prefix = f"m_{date}_"
        count = sum(1 for e in self.entries() if e.id.startswith(prefix))
        return f"{prefix}{count + 1:04d}"

    def _read_lines(self) -> list[tuple[str, SemanticEntry | None]]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        result: list[tuple[str, SemanticEntry | None]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                entry = SemanticEntry.from_dict(data) if isinstance(data, dict) else None
            except json.JSONDecodeError:
                entry = None
            if entry is None:
                _log.warning("semantic store %s: skipping unreadable line", self.path)
            result.append((line, entry))
        return result

    def _schedule_commit(self, kind: str) -> None:
        if self.commit_debounce_sec is None:
            return
        schedule_commit(self.path, self.persona_stem, kind, debounce_sec=self.commit_debounce_sec)
