"""mltgnt.memory — re-export hub."""
from mltgnt.memory._commit import flush_memory_commits
from mltgnt.memory._format import (
    MemoryEntry,
    assemble_entries_text,
    parse_jsonl,
    serialize_entry,
)
from mltgnt.memory._chroma import get_collection, query_similar, upsert_entry
from mltgnt.memory.api import (
    MEMORY_CORRUPT_THRESHOLD_BYTES,
    MEMORY_DEDUPE_SCAN_BYTES,
    MEMORY_DEDUPE_SCAN_LINES,
    _ensure_jsonl,
    _resolve_memory_dir,
    _scan_tail_for_dedupe_key,
    append_memory_entry,
    memory_file_path,
    persona_memory_lock,
    read_memory_preferences,
    read_memory_tail_text,
    tail_utf8_bytes,
)
from mltgnt.memory.search import (
    _search_and_score,
    read_memory_by_relevance,
    read_memory_iterative,
    read_memory_with_sufficiency_check,
)
from mltgnt.memory.compaction import (
    CompactionResult,
    LlmCall,
    LlmCallError,
    compact,
    needs_compaction,
)
from mltgnt.memory.archive import archive_episodes
from mltgnt.memory.core_render import render_core
from mltgnt.memory.reflection import (
    ApplyReport,
    ReflectionAdd,
    ReflectionParseError,
    ReflectionResult,
    apply_reflection,
    build_reflection_prompt,
    parse_reflection,
)
from mltgnt.memory.semantic import KINDS, SemanticEntry, SemanticStore, validate_entry
from mltgnt.memory.tools import MEMORY_TOOL_SPECS, MemoryGate, MemoryToolExecutor

# The semantic-memory API above is importable from ``mltgnt.memory`` but kept out
# of ``__all__`` (pinned by tests/test_all_snapshot.py); its submodules
# (semantic / core_render / tools / reflection / archive) declare their own ``__all__``.
_SEMANTIC_API = (
    ApplyReport,
    KINDS,
    MEMORY_TOOL_SPECS,
    MemoryGate,
    MemoryToolExecutor,
    ReflectionAdd,
    ReflectionParseError,
    ReflectionResult,
    SemanticEntry,
    SemanticStore,
    apply_reflection,
    archive_episodes,
    build_reflection_prompt,
    parse_reflection,
    render_core,
    validate_entry,
)

__all__ = [
    "persona_memory_lock",
    "append_memory_entry",
    "read_memory_preferences",
    "read_memory_tail_text",
    "read_memory_by_relevance",
    "read_memory_with_sufficiency_check",
    "read_memory_iterative",
    "memory_file_path",
    "LlmCallError",
    "CompactionResult",
    "compact",
    "flush_memory_commits",
    "needs_compaction",
    "MemoryEntry",
    "parse_jsonl",
    "serialize_entry",
    "assemble_entries_text",
    "get_collection",
    "query_similar",
    "upsert_entry",
    "tail_utf8_bytes",
    "LlmCall",
    "MEMORY_CORRUPT_THRESHOLD_BYTES",
    "MEMORY_DEDUPE_SCAN_BYTES",
    "MEMORY_DEDUPE_SCAN_LINES",
    "_ensure_jsonl",
    "_resolve_memory_dir",
    "_scan_tail_for_dedupe_key",
    "_search_and_score",
]
