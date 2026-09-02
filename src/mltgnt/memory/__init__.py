"""mltgnt.memory — re-export hub."""
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
    _tail_utf8_bytes,
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
    "_tail_utf8_bytes",
    "_search_and_score",
]
