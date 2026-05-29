"""全公開モジュールの __all__ スナップショットテスト（Issue #1300 P1-1）。"""
from __future__ import annotations

import importlib

import pytest

EXPECTED_ALL: dict[str, list[str]] = {
    "mltgnt": [
        "AgentResult",
        "AgentRunner",
        "ChatInput",
        "ChatOutput",
        "Message",
        "Persona",
        "PersonaProtocol",
        "PersonaScheduler",
        "ScheduleJob",
        "__version__",
        "compact",
        "enqueue_and_wait",
        "enqueue_dag",
        "list_personas",
        "load_persona",
        "needs_compaction",
        "read_memory_by_relevance",
        "read_memory_iterative",
        "read_memory_with_sufficiency_check",
        "run_persona_prompt",
        "run_pipeline",
        "validate_persona",
    ],
    "mltgnt.memory": [
        "CompactionResult",
        "LlmCall",
        "LlmCallError",
        "MEMORY_CORRUPT_THRESHOLD_BYTES",
        "MEMORY_DEDUPE_SCAN_BYTES",
        "MEMORY_DEDUPE_SCAN_LINES",
        "MemoryEntry",
        "_ensure_jsonl",
        "_resolve_memory_dir",
        "_scan_tail_for_dedupe_key",
        "_search_and_score",
        "_tail_utf8_bytes",
        "append_memory_entry",
        "assemble_entries_text",
        "compact",
        "memory_file_path",
        "needs_compaction",
        "parse_jsonl",
        "persona_memory_lock",
        "read_memory_by_relevance",
        "read_memory_iterative",
        "read_memory_preferences",
        "read_memory_tail_text",
        "read_memory_with_sufficiency_check",
        "serialize_entry",
        "tail_utf8_bytes",
    ],
    "mltgnt.skill": [
        "ArtifactSpec",
        "ConsumesSpec",
        "ProducesSpec",
        "RunOutput",
        "SkillFile",
        "SkillMatchResult",
        "SkillMeta",
        "SkillRegistry",
        "SkillRunResult",
        "discover",
        "discover_bodies",
        "lint_skill_meta",
        "load",
        "match",
        "resolve_skill",
        "run",
    ],
    "mltgnt.agent": [
        "AgentResult",
        "AgentRunner",
    ],
    "mltgnt.chat": [
        "ChatInput",
        "ChatOutput",
        "Message",
        "run_pipeline",
    ],
    "mltgnt.bridges": [
        "DagStep",
        "MltgntHooks",
        "call_llm",
        "create_audit_writer",
        "enqueue_and_wait",
        "enqueue_dag",
        "files_adapter",
        "ghdag_bridge",
        "hooks_adapter",
        "llm_adapter",
        "md_read",
        "md_write",
    ],
    "mltgnt.routing": [
        "ChannelPersonaEntry",
        "RoutingRule",
        "TRIAGE_PROFILE_MAX_CHARS",
        "detect_nickname",
        "evaluate",
        "extract_json_object",
        "extract_triage_section",
        "find_observers",
        "load_channel_persona_map",
        "prepare_profile_for_triage",
        "resolve_responding_persona",
    ],
    "mltgnt.persona": [
        "Persona",
        "PersonaValidationError",
        "compress_heavy_to_light",
        "list_personas",
        "load_persona",
        "regenerate_light_block",
        "run_persona_prompt",
        "validate_persona",
    ],
    "mltgnt.interfaces": [
        "ChatInput",
        "ChatInputBase",
        "ChatOutput",
        "ChatOutputBase",
        "ChatPipelineProtocol",
        "Message",
        "PersonaFMBase",
        "PersonaProtocol",
        "SlackClientProtocol",
    ],
    "mltgnt.scheduler": [
        "PersonaScheduler",
        "ScheduleJob",
        "SchedulePaths",
        "_hash_offset",
        "atomic_write_text",
        "load_schedule_jobs",
    ],
    "mltgnt.daemon": [
        "DaemonComponent",
        "DaemonRunner",
        "PidLock",
        "SkillWatcherComponent",
    ],
    "mltgnt.config": [
        "ChatConfig",
        "DEFAULT_WEIGHT_MAP",
        "MemoryConfig",
        "PersonaConfig",
        "SchedulerConfig",
    ],
    "mltgnt.exceptions": [
        "ConfigError",
        "DependencyError",
        "MltgntError",
    ],
}


@pytest.mark.parametrize("module_path,expected", EXPECTED_ALL.items())
def test_all_snapshot(module_path: str, expected: list[str]) -> None:
    mod = importlib.import_module(module_path)
    actual = sorted(getattr(mod, "__all__", []))
    assert actual == sorted(expected), (
        f"{module_path}.__all__ changed.\n"
        f"  Added: {set(actual) - set(expected)}\n"
        f"  Removed: {set(expected) - set(actual)}"
    )


def test_all_snapshot_covers_thirteen_modules() -> None:
    assert len(EXPECTED_ALL) == 13
