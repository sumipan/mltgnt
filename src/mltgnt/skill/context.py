"""mltgnt.skill.context — build extra_context from knowledge + memory.

Design: Issue #3030 / #3173
"""
from __future__ import annotations

import json
from pathlib import Path

from mltgnt.skill.models import SkillMeta


def build_extra_context(
    skill_meta: SkillMeta,
    repo_root: Path,
    persona_name: str,
    *,
    knowledge_count: int = 0,
    memory_max_bytes: int = 0,
    memory_exclude_source_tags: list[str] | None = None,
) -> str | None:
    """Build an extra_context string from knowledge and memory files.

    Args:
        skill_meta: SkillMeta.knowledge_paths must hold indexed paths
        repo_root: Repo root (for resolving memory files)
        persona_name: Persona name (used in memory filenames)
        knowledge_count: Paragraphs from the end (blank-line delimited). Default 0 (off)
        memory_max_bytes: Memory bytes from the end. Default 0 (off)
        memory_exclude_source_tags: Exclude JSONL rows with matching source_tag (exact)

    Returns:
        None when both knowledge and memory are empty (extra_context=None compat)
        Otherwise a string like "### knowledge (last N)\n\n..."
    """
    knowledge_text = _read_knowledge(skill_meta.knowledge_paths, knowledge_count)
    memory_text = _read_memory(
        repo_root,
        persona_name,
        memory_max_bytes,
        exclude_source_tags=memory_exclude_source_tags,
    )

    parts: list[str] = []
    if knowledge_text:
        parts.append(f"### knowledge (last {knowledge_count})\n\n{knowledge_text}")
    if memory_text:
        parts.append(f"### Memory (tail)\n\n{memory_text}")
    return "\n\n".join(parts) if parts else None


def _read_knowledge(knowledge_paths: list[Path], knowledge_count: int) -> str:
    """Concatenate knowledge_paths files; return the last knowledge_count paragraphs."""
    if not knowledge_paths or knowledge_count <= 0:
        return ""
    texts: list[str] = []
    for p in knowledge_paths:
        if p.is_file():
            texts.append(p.read_text(encoding="utf-8"))
    combined = "\n\n".join(texts)
    paragraphs = [seg for seg in combined.split("\n\n") if seg.strip()]
    return "\n\n".join(paragraphs[-knowledge_count:])


def _read_memory(
    repo_root: Path,
    persona_name: str,
    max_bytes: int,
    *,
    exclude_source_tags: list[str] | None = None,
) -> str:
    """Read the last max_bytes of a persona memory file; format JSONL as bullets."""
    memory_file = repo_root / "chat" / "memory" / f"{persona_name}.jsonl"
    if not memory_file.is_file() or max_bytes <= 0:
        return ""
    file_size = memory_file.stat().st_size
    start = max(0, file_size - max_bytes)
    with memory_file.open("rb") as f:
        f.seek(start)
        data = f.read(max_bytes)
    text = data.decode("utf-8", errors="replace")
    if start > 0:
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
    text = text.lstrip("\n")
    if not text.strip():
        return ""

    exclude = set(exclude_source_tags or [])
    lines_out: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        content = record.get("content")
        if not isinstance(content, str) or not content:
            continue
        source_tag = record.get("source_tag")
        if exclude and isinstance(source_tag, str) and source_tag in exclude:
            continue
        timestamp = record.get("timestamp", "")
        role = record.get("role", "")
        if not isinstance(timestamp, str):
            timestamp = str(timestamp) if timestamp is not None else ""
        if not isinstance(role, str):
            role = str(role) if role is not None else ""
        lines_out.append(f"- [{timestamp}] {role}: {content}")
    return "\n".join(lines_out)
