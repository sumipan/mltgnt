"""mltgnt.memory.dream._format — DreamSection / DreamSummary と JSON 変換。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DreamSection",
    "DreamSummary",
    "dream_summary_to_dict",
    "dream_summary_from_dict",
]


@dataclass(frozen=True)
class DreamSection:
    category: str
    content: str
    source_entries: int


@dataclass(frozen=True)
class DreamSummary:
    persona: str
    sections: list[DreamSection]
    updated_at: str


def dream_summary_to_dict(summary: DreamSummary) -> dict[str, Any]:
    return {
        "persona": summary.persona,
        "updated_at": summary.updated_at,
        "sections": [
            {
                "category": s.category,
                "content": s.content,
                "source_entries": s.source_entries,
            }
            for s in summary.sections
        ],
    }


def dream_summary_from_dict(data: dict[str, Any]) -> DreamSummary:
    sections_raw = data.get("sections") or []
    sections: list[DreamSection] = []
    for item in sections_raw:
        if not isinstance(item, dict):
            continue
        sections.append(DreamSection(
            category=str(item.get("category", "")),
            content=str(item.get("content", "")),
            source_entries=int(item.get("source_entries", 0)),
        ))
    return DreamSummary(
        persona=str(data.get("persona", "")),
        sections=sections,
        updated_at=str(data.get("updated_at", "")),
    )


def dream_summary_to_json(summary: DreamSummary) -> str:
    return json.dumps(dream_summary_to_dict(summary), ensure_ascii=False, indent=2) + "\n"


def dream_summary_from_json(text: str) -> DreamSummary:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("dream.json root must be an object")
    return dream_summary_from_dict(data)
