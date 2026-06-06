"""mltgnt.memory.dream.api — dream.json の読み書き。"""
from __future__ import annotations

import json
from pathlib import Path

from mltgnt.config import MemoryConfig
from mltgnt.memory.dream._format import (
    DreamSummary,
    dream_summary_from_json,
    dream_summary_to_json,
)

__all__ = ["read_dream", "read_dream_summary", "write_dream", "dream_json_path"]


def dream_json_path(persona_dir: Path, *, memory_dir_name: str = "memory") -> Path:
    return persona_dir / memory_dir_name / "dream.json"


def read_dream(
    persona_dir: Path,
    *,
    memory_dir_name: str = "memory",
) -> DreamSummary | None:
    path = dream_json_path(persona_dir, memory_dir_name=memory_dir_name)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        return dream_summary_from_json(text)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None


def write_dream(
    persona_dir: Path,
    summary: DreamSummary,
    *,
    memory_dir_name: str = "memory",
) -> None:
    path = dream_json_path(persona_dir, memory_dir_name=memory_dir_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dream_summary_to_json(summary), encoding="utf-8")
    tmp.replace(path)


def read_dream_summary(config: MemoryConfig, persona_stem: str) -> str:
    """dream.json をカテゴリ見出し付きテキストにフォーマットして返す。"""
    persona_dir = config.chat_dir / persona_stem
    summary = read_dream(persona_dir, memory_dir_name=config.dream_dir_name)
    if summary is None or not summary.sections:
        return ""

    lines = ["", "", "## 記憶の要約", ""]
    for section in summary.sections:
        lines.append(f"### {section.category}")
        lines.append(section.content)
        lines.append("")
    return "\n".join(lines).rstrip("\n")
