from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ALLOWED_CATEGORIES = {"skill_mismatch", "persona_quality", "triage_error", "timeout"}


@dataclass
class FailurePattern:
    pattern_id: str
    category: str
    count: int
    example_correlation_ids: list[str]
    affected_persona: str | None
    affected_skill: str | None


def _extract_date(record: dict) -> date | None:
    raw = record.get("timestamp")
    if not isinstance(raw, str) or len(raw) < 10:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def analyze_failures(
    audit_path: Path,
    *,
    since: date | None = None,
    until: date | None = None,
) -> list[FailurePattern]:
    """audit.jsonl から失敗パターンを抽出・集約する。"""
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []

    records: list[dict] = []
    for line in lines:
        if not line.strip():
            continue
        records.append(json.loads(line))

    exit_counts: Counter[str] = Counter()
    for record in records:
        if record.get("event_type") == "task_exit":
            corr = record.get("correlation_id")
            if isinstance(corr, str) and corr:
                exit_counts[corr] += 1

    grouped: dict[tuple[str, str, str | None, str | None], list[str]] = defaultdict(list)
    for record in records:
        if record.get("event_type") != "task_failed":
            continue

        record_date = _extract_date(record)
        if since is not None and record_date is not None and record_date < since:
            continue
        if until is not None and record_date is not None and record_date > until:
            continue

        corr = record.get("correlation_id")
        if not isinstance(corr, str) or not corr:
            continue

        skill = record.get("skill")
        persona = record.get("persona")
        skill_name = skill if isinstance(skill, str) and skill else None
        persona_name = persona if isinstance(persona, str) and persona else None

        error = record.get("error")
        error_text = error.lower() if isinstance(error, str) else ""

        if corr.startswith("slack:") and exit_counts[corr] >= 2:
            category = "triage_error"
        elif "timeout" in error_text:
            category = "timeout"
        elif skill_name is not None:
            category = "skill_mismatch"
        else:
            category = "persona_quality"

        if category not in ALLOWED_CATEGORIES:
            continue
        pattern_id = f"{category}:{skill_name or persona_name or corr}"
        grouped[(pattern_id, category, persona_name, skill_name)].append(corr)

    patterns: list[FailurePattern] = []
    for (pattern_id, category, persona_name, skill_name), corr_ids in grouped.items():
        examples: list[str] = []
        for corr in corr_ids:
            if corr not in examples:
                examples.append(corr)
            if len(examples) == 3:
                break
        patterns.append(
            FailurePattern(
                pattern_id=pattern_id,
                category=category,
                count=len(corr_ids),
                example_correlation_ids=examples,
                affected_persona=persona_name,
                affected_skill=skill_name,
            )
        )

    patterns.sort(key=lambda item: (-item.count, item.pattern_id))
    return patterns
