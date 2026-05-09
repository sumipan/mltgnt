"""
mltgnt.config — diary 非依存な設定 dataclass。

設計: Issue #118 §4.1, Issue #123 §4.1
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "MemoryConfig",
    "SchedulerConfig",
    "ChatConfig",
]


@dataclass(frozen=True)
class MemoryConfig:
    """メモリ管理に必要なパス・閾値。"""
    chat_dir: Path
    chat_memory_dir: Path | None = None
    inject_max_bytes: int = 10_240
    inject_max_entries: int = 12
    preferences_max_bytes: int = 5_120
    lock_timeout_sec: float = 30.0
    raw_days: int = 7
    mid_weeks: int = 3
    compact_threshold_bytes: int = 40_960
    compact_target_bytes: int = 25_600
    preferences_section_name: str = "ユーザーの好み・傾向"
    protected_layers: tuple[str, ...] = ("caveat",)


@dataclass(frozen=True)
class SchedulerConfig:
    """スケジューラに必要なパス・設定。"""
    schedule_yaml: Path
    state_dir: Path
    timezone: str = "Asia/Tokyo"
    salt: str = ""


@dataclass(frozen=True)
class ChatConfig:
    """チャットパイプラインに必要な設定。"""
    persona_dir: Path
    memory_dir: Path | None = None
    sufficiency_engine: str | None = None   # 例: "claude"
    sufficiency_model: str | None = None    # 例: "claude-haiku-4-5-20251001"
