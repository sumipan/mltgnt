"""mltgnt.memory — re-export hub."""
from mltgnt.memory._format import (
    MemoryEntry,
    assemble_entries_text,
    parse_jsonl,
    serialize_entry,
)
from mltgnt.memory.api import (
    LlmCall,
    MEMORY_CORRUPT_THRESHOLD_BYTES,
    MEMORY_DEDUPE_SCAN_BYTES,
    MEMORY_DEDUPE_SCAN_LINES,
    _ensure_jsonl,
    _resolve_memory_dir,
    _scan_tail_for_dedupe_key,
    _tail_utf8_bytes,
    append_memory_entry,
    memory_file_path,
    normalize_source_prefix,
    persona_memory_lock,
    read_memory_preferences,
    read_memory_tail_text,
    tail_utf8_bytes,
)
from mltgnt.memory.search import (
    _search_and_score,
    read_memory_agentic,
    read_memory_by_relevance,
    read_memory_iterative,
    read_memory_with_sufficiency_check,
)
from mltgnt.memory.compaction import (
    CompactionResult,
    LlmCallError,
    compact,
    needs_compaction,
)

__all__ = [
    "persona_memory_lock",
    "append_memory_entry",
    "read_memory_preferences",
    "read_memory_tail_text",
    "read_memory_by_relevance",
    "read_memory_with_sufficiency_check",
    "read_memory_iterative",
    "read_memory_agentic",
    "memory_file_path",
    "normalize_source_prefix",
    "compact",
    "needs_compaction",
    "LlmCallError",
    "CompactionResult",
    "MemoryEntry",
    "parse_jsonl",
    "serialize_entry",
    "assemble_entries_text",
    "tail_utf8_bytes",
]
