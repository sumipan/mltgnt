"""
mltgnt.config — diary-independent configuration dataclasses.

Design: Issue #118 §4.1, Issue #123 §4.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mltgnt.config.language import JA, LanguagePack

__all__ = [
    "DEFAULT_WEIGHT_MAP",
    "ConversationConfig",
    "MemoryConfig",
    "PersonaConfig",
    "SchedulerConfig",
]

DEFAULT_WEIGHT_MAP: dict[str, str] = {
    # Japanese text intentionally kept for CJK processing test
    # Japanese (v1/v2) persona YAML section names
    "基本情報": "heavy",
    "価値観": "heavy",
    "反応パターン": "heavy",
    # Japanese text intentionally kept for CJK processing test
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
    """Settings required for persona load and interpretation."""
    weight_map: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHT_MAP)
    )
    exclude_stems: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ConversationConfig:
    """Conversation-layer storage paths and thresholds (queue, session, ledger).

    Like MemoryConfig, the host injects paths. Do not hardcode them.
    """

    queue_dir: Path
    sessions_dir: Path
    ledger_dir: Path
    thread_index_dir: Path
    thread_persona_path: Path
    posts_dir: Path | None = None
    audit_path: Path | None = None
    stale_after_sec: int = 3600
    max_queued: int = 20
    cleanup_ttl_days: int = 14
    thread_persona_ttl_days: int = 30


@dataclass(frozen=True)
class MemoryConfig:
    """Paths and thresholds for memory management."""
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
    # Japanese text intentionally kept for CJK processing test
    preferences_section_name: str = "ユーザーの好み・傾向"
    protected_layers: tuple[str, ...] = ("caveat",)
    timezone: str = "Asia/Tokyo"  # used by _redistribute_entries
    dream_model: str = "claude-haiku-4-5-20251001"
    use_dream_summary: bool = False
    dream_dir_name: str = "memory"
    global_dream_exclude_personas: tuple[str, ...] = ()


@dataclass(frozen=True)
class SchedulerConfig:
    """Paths and settings for the scheduler."""
    schedule_yaml: Path
    state_dir: Path
    timezone: str = "Asia/Tokyo"
    salt: str = ""
