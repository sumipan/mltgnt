"""
mltgnt.config — diary-independent configuration dataclasses.

Design: Issue #118 §4.1, Issue #123 §4.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, overload

from mltgnt.config.language import JA, LanguagePack

__all__ = [
    "DEFAULT_WEIGHT_MAP",
    "ConversationConfig",
    "MemoryConfig",
    "PersonaConfig",
    "SchedulerConfig",
]

_T = TypeVar("_T")


class _CanonicalWeightMap(dict[str, str]):
    """English-keyed mapping with lookup-only support for legacy headings."""

    @overload
    def get(self, key: str, default: None = None, /) -> str | None: ...

    @overload
    def get(self, key: str, default: str, /) -> str: ...

    @overload
    def get(self, key: str, default: _T, /) -> str | _T: ...

    def get(self, key: str, default: _T | None = None, /) -> str | _T | None:
        canonical = PERSONA_SECTION_ALIASES.get(key, key)
        return super().get(canonical, default)


# Assigned below after the legacy alias table is available.
DEFAULT_WEIGHT_MAP: dict[str, str]

# Legacy headings are decoded by Python while keeping locale data out of source
# text. Consumers normalize these aliases to canonical English keys.
PERSONA_SECTION_ALIASES: dict[str, str] = {
    "\N{CJK UNIFIED IDEOGRAPH-57FA}\N{CJK UNIFIED IDEOGRAPH-672C}\N{CJK UNIFIED IDEOGRAPH-60C5}\N{CJK UNIFIED IDEOGRAPH-5831}": "Background",
    "\N{CJK UNIFIED IDEOGRAPH-4FA1}\N{CJK UNIFIED IDEOGRAPH-5024}\N{CJK UNIFIED IDEOGRAPH-89B3}": "Values",
    "\N{CJK UNIFIED IDEOGRAPH-53CD}\N{CJK UNIFIED IDEOGRAPH-5FDC}\N{KATAKANA LETTER PA}\N{KATAKANA LETTER TA}\N{KATAKANA-HIRAGANA PROLONGED SOUND MARK}\N{KATAKANA LETTER N}": "Reaction patterns",
    "\N{CJK UNIFIED IDEOGRAPH-53E3}\N{CJK UNIFIED IDEOGRAPH-8ABF}": "Tone",
    "\N{KATAKANA LETTER A}\N{KATAKANA LETTER U}\N{KATAKANA LETTER TO}\N{KATAKANA LETTER PU}\N{KATAKANA LETTER SMALL TU}\N{KATAKANA LETTER TO}\N{CJK UNIFIED IDEOGRAPH-5F62}\N{CJK UNIFIED IDEOGRAPH-5F0F}": "Output format",
    "\N{CJK UNIFIED IDEOGRAPH-8EFD}\N{CJK UNIFIED IDEOGRAPH-91CF}": "Light",
    "\N{CJK UNIFIED IDEOGRAPH-91CD}\N{CJK UNIFIED IDEOGRAPH-91CF}": "Heavy",
    "\N{CJK UNIFIED IDEOGRAPH-53C2}\N{CJK UNIFIED IDEOGRAPH-7167}": "Reference",
    "\N{KATAKANA LETTER TO}\N{KATAKANA LETTER RI}\N{KATAKANA LETTER A}\N{KATAKANA-HIRAGANA PROLONGED SOUND MARK}\N{KATAKANA LETTER ZI}\N{CJK UNIFIED IDEOGRAPH-7528}": "Triage",
}

DEFAULT_WEIGHT_MAP = _CanonicalWeightMap(
    {
        "Background": "heavy",
        "Values": "heavy",
        "Reaction patterns": "heavy",
        "Tone": "heavy",
        "Output format": "reference",
        "Light": "light",
    }
)


@dataclass(frozen=True)
class PersonaConfig:
    """Settings required for persona load and interpretation."""

    weight_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_WEIGHT_MAP))
    section_aliases: dict[str, str] = field(default_factory=lambda: dict(PERSONA_SECTION_ALIASES))
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
    preferences_section_name: str = "User’s preferences and tendencies"
    protected_layers: tuple[str, ...] = ("caveat",)
    timezone: str = "Asia/Tokyo"  # used by _redistribute_entries
    dream_model: str = ""  # empty: engine default (claude falls back to haiku)
    dream_engine: str = "claude"  # "claude" / "cursor" / "codex"
    use_dream_summary: bool = False
    dream_dir_name: str = "memory"
    commit_debounce_sec: float = 300.0  # memory git commit debounce (nexus #3833)
    global_dream_exclude_personas: tuple[str, ...] = ()


@dataclass(frozen=True)
class SchedulerConfig:
    """Paths and settings for the scheduler."""

    schedule_yaml: Path
    state_dir: Path
    timezone: str = "Asia/Tokyo"
    salt: str = ""
