"""
mltgnt.config — diary 非依存な設定 dataclass。

設計: Issue #118 §4.1, Issue #123 §4.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

__all__ = [
    "DEFAULT_WEIGHT_MAP",
    "MemoryConfig",
    "PersonaConfig",
    "SchedulerConfig",
    "ChatConfig",
]

DEFAULT_WEIGHT_MAP: dict[str, str] = {
    # 日本語（v1/v2）
    "基本情報": "heavy",
    "価値観": "heavy",
    "反応パターン": "heavy",
    "口調": "heavy",
    "アウトプット形式": "reference",
    "軽量": "light",
    # English
    "Background": "heavy",
    "Values": "heavy",
    "Tone": "heavy",
    "Output format": "reference",
    "Light": "light",
}


@dataclass(frozen=True)
class PersonaConfig:
    """ペルソナ読み込み・解釈に必要な設定。"""
    weight_map: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHT_MAP)
    )


@dataclass(frozen=True)
class MemoryConfig:
    """メモリ管理に必要なパス・閾値。"""
    chat_dir: Path
    chat_memory_dir: Path | None = None
    inject_max_bytes: int = 10_240
    inject_max_entries: int = 12
    preferences_max_bytes: int = 5_120
    lock_timeout_sec: float = 30.0
    lock_stale_threshold_sec: float = 300.0
    raw_days: int = 7
    mid_weeks: int = 3
    compact_threshold_bytes: int = 40_960
    compact_target_bytes: int = 25_600
    preferences_section_name: str = "ユーザーの好み・傾向"
    protected_layers: tuple[str, ...] = ("caveat",)
    timezone: str = "Asia/Tokyo"  # _redistribute_entries で使用
    dream_model: str = "claude-haiku-4-5-20251001"
    use_dream_summary: bool = False
    dream_dir_name: str = "memory"
    global_dream_exclude_personas: tuple[str, ...] = ()


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
    matcher_model: str = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class LoopsConfig:
    """loops コンポーネントに必要なパス・閾値。"""
    objectives_dir: Path
    state_dir: Path
    status_dir: Path
    jobs_dir: Path
    exec_done_dir: Path
    persona_dir: Path
    default_persona: str
    fallback_channel: str
    poll_interval_sec: float = 10.0
    max_iterations: int = 5
    max_clarify_rounds: int = 3
    max_subtasks_per_iteration: int = 5
    subtask_timeout_sec: float = 1800.0
    llm_engine: str = "claude"
    llm_model: str = ""
    subtask_engine: str = "claude"
    subtask_model: str = ""
    on_status_written: Callable[[Path], None] | None = None
    progress_notify: bool = True
    deliverable_excerpt_chars: int = 4000
    result_summary_chars: int = 1000
    watch_root: Path | None = None
    max_replans_per_iteration: int = 3
    max_plan_revisions: int = 3
    plan_approval_default: bool = True

    def __post_init__(self) -> None:
        if self.poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be positive")
        if self.subtask_timeout_sec <= 0:
            raise ValueError("subtask_timeout_sec must be positive")
        if not (1 <= self.max_iterations <= 10):
            raise ValueError("max_iterations must be in 1..10")
        if not (1 <= self.max_clarify_rounds <= 3):
            raise ValueError("max_clarify_rounds must be in 1..3")
        if not (1 <= self.max_subtasks_per_iteration <= 5):
            raise ValueError("max_subtasks_per_iteration must be in 1..5")
        if not self.default_persona.strip():
            raise ValueError("default_persona must not be empty")
        if self.deliverable_excerpt_chars <= 0:
            raise ValueError("deliverable_excerpt_chars must be positive")
        if self.result_summary_chars <= 0:
            raise ValueError("result_summary_chars must be positive")
        if not (0 <= self.max_replans_per_iteration <= 10):
            raise ValueError("max_replans_per_iteration must be in 0..10")
        if not (0 <= self.max_plan_revisions <= 10):
            raise ValueError("max_plan_revisions must be in 0..10")
