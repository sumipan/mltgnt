"""mltgnt.memory.reflect_job — deterministic collect / validate / apply steps of the reflection job."""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from mltgnt.memory._format import MemoryEntry, parse_jsonl
from mltgnt.memory.archive import archive_episodes
from mltgnt.memory.reflection import (
    ReflectionParseError,
    apply_reflection,
    parse_reflection,
)
from mltgnt.memory.semantic import SemanticStore

__all__ = [
    "COLLECT_VERSION",
    "LLM_VERSION",
    "apply_from_payloads",
    "build_collect_payload",
    "collect_new_episodes",
    "load_reflect_state",
    "save_reflect_state",
    "validate_llm_payload",
]

COLLECT_VERSION = 1
LLM_VERSION = 1

_log = logging.getLogger(__name__)


def load_reflect_state(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_reflect_state(path: Path, state: dict[str, dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _persona_dirs(memory_dir: Path) -> list[Path]:
    if not memory_dir.is_dir():
        return []
    return sorted(
        d for d in memory_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "archive"
    )


def collect_new_episodes(persona_dir: Path, last_ts: str | None) -> list[MemoryEntry]:
    ep_path = persona_dir / "episodes.jsonl"
    if not ep_path.exists():
        return []
    all_entries = parse_jsonl(ep_path)
    if last_ts is None:
        return all_entries
    return [e for e in all_entries if e.timestamp > last_ts]


def build_collect_payload(
    memory_dir: Path,
    *,
    state_file: Path,
    now: datetime,
    persona_filter: Sequence[str] | None = None,
) -> dict[str, Any]:
    state = load_reflect_state(state_file)
    items: list[dict[str, Any]] = []
    for persona_dir in _persona_dirs(memory_dir):
        stem = persona_dir.name
        if persona_filter and stem not in persona_filter:
            continue
        archive_episodes(persona_dir, now=now)
        last_ts = (state.get(stem) or {}).get("last_ts")
        episodes = collect_new_episodes(persona_dir, last_ts)
        if not episodes:
            continue
        store = SemanticStore(persona_dir / "semantic.jsonl", persona_stem=stem)
        active = store.active()
        items.append(
            {
                "persona": stem,
                "episodes": [asdict(e) for e in episodes],
                "active": [asdict(a) for a in active],
            }
        )
    return {"version": COLLECT_VERSION, "items": items}


def validate_llm_payload(
    collect: dict[str, Any],
    llm: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if llm.get("version") != LLM_VERSION:
        raise ValueError("llm version mismatch")
    if collect.get("version") != COLLECT_VERSION:
        raise ValueError("collect version mismatch")

    need = sorted(
        str(i["persona"])
        for i in collect.get("items") or []
        if isinstance(i, dict) and i.get("persona")
    )
    llm_items = llm.get("items")
    if not isinstance(llm_items, list):
        raise ValueError("llm.items must be a list")

    by_persona: dict[str, dict[str, Any]] = {}
    for row in llm_items:
        if not isinstance(row, dict):
            raise ValueError("llm item must be object")
        persona = row.get("persona")
        if not isinstance(persona, str) or not persona:
            raise ValueError("persona required")
        if persona in by_persona:
            raise ValueError(f"duplicate persona: {persona}")
        by_persona[persona] = row

    got = sorted(by_persona)
    if got != need:
        raise ValueError(f"persona set mismatch: expected {need}, got {got}")
    return by_persona


def _apply_one_persona(
    stem: str,
    persona_dir: Path,
    persona_state: dict[str, Any],
    llm_row: dict[str, Any],
    *,
    collect_row: dict[str, Any],
    now: datetime,
    dry_run: bool,
    failure_threshold: int,
    on_failure: Callable[[str, int, str], None] | None,
) -> tuple[bool, dict[str, Any]]:
    last_ts = persona_state.get("last_ts")
    episodes_raw = collect_row.get("episodes") or []
    if not episodes_raw:
        return True, {"last_ts": last_ts, "failures": 0}

    if dry_run:
        _log.info("dry-run: would reflect %d episodes for %s", len(episodes_raw), stem)
        return True, persona_state

    store = SemanticStore(persona_dir / "semantic.jsonl", persona_stem=stem)
    run_id = f"{now.strftime('%Y%m%dT%H%M')}-{stem}"
    ts = now.isoformat()
    reflection_body = json.dumps(
        {
            "reflection": llm_row.get("reflection", ""),
            "add": llm_row.get("add") or [],
            "supersede": llm_row.get("supersede") or [],
        },
        ensure_ascii=False,
    )

    try:
        reflection = parse_reflection(reflection_body)
        apply_reflection(store, reflection, run_id=run_id, ts=ts)
        timestamps = [
            str(e.get("timestamp") or "")
            for e in episodes_raw
            if isinstance(e, dict) and e.get("timestamp")
        ]
        new_ts = max(timestamps) if timestamps else last_ts
        _log.info("reflected %s: %d episodes", stem, len(episodes_raw))
        return True, {"last_ts": new_ts, "failures": 0}

    except (ReflectionParseError, Exception) as e:
        failures = int(persona_state.get("failures", 0)) + 1
        err_line = str(e).splitlines()[0] if str(e) else "unknown error"
        _log.error("reflect failed for %s (failures=%d): %s", stem, failures, err_line)
        if failures == failure_threshold and on_failure is not None:
            on_failure(stem, failures, err_line)
        return False, {"last_ts": last_ts, "failures": failures}


def apply_from_payloads(
    collect: dict[str, Any],
    llm: dict[str, Any],
    *,
    memory_dir: Path,
    state_file: Path,
    now: datetime,
    dry_run: bool = False,
    failure_threshold: int = 3,
    on_failure: Callable[[str, int, str], None] | None = None,
) -> tuple[bool, str | None]:
    try:
        by_persona = validate_llm_payload(collect, llm)
    except ValueError as e:
        return False, str(e)

    state = load_reflect_state(state_file)
    any_failure = False
    collect_by_stem = {
        str(row["persona"]): row
        for row in collect.get("items") or []
        if isinstance(row, dict) and row.get("persona")
    }

    for stem, llm_row in by_persona.items():
        persona_dir = memory_dir / stem
        persona_state = state.get(stem, {"last_ts": None, "failures": 0})
        success, updated = _apply_one_persona(
            stem,
            persona_dir,
            persona_state,
            llm_row,
            collect_row=collect_by_stem[stem],
            now=now,
            dry_run=dry_run,
            failure_threshold=failure_threshold,
            on_failure=on_failure,
        )
        state[stem] = updated
        if not success:
            any_failure = True

    if not dry_run:
        save_reflect_state(state_file, state)

    return not any_failure, None
