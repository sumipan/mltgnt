"""mltgnt.memory.dream.selector — 合成対象 persona の選別。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mltgnt.memory.dream.api import read_dream

__all__ = ["DreamSelector"]


class DreamSelector:
    @staticmethod
    def pick_targets(
        persona_dirs: list[Path],
        *,
        memory_dir_name: str = "memory",
    ) -> list[Path]:
        targets: list[Path] = []
        for persona_dir in persona_dirs:
            memory_dir = persona_dir / memory_dir_name
            jsonl_files = list(memory_dir.glob("*.jsonl"))
            if not jsonl_files:
                continue
            latest_mtime = max(f.stat().st_mtime for f in jsonl_files)
            existing = read_dream(persona_dir, memory_dir_name=memory_dir_name)
            updated_epoch = _dream_updated_epoch(existing)
            if latest_mtime > updated_epoch:
                targets.append(persona_dir)
        return targets


def _dream_updated_epoch(summary) -> float:
    if summary is None:
        return 0.0
    try:
        dt = datetime.fromisoformat(summary.updated_at)
        return dt.timestamp()
    except ValueError:
        return 0.0
