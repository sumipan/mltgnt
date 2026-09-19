"""Unit tests for the Chroma semantic search backend."""
from __future__ import annotations

from pathlib import Path

import pytest

chromadb = pytest.importorskip("chromadb")

from mltgnt.config import MemoryConfig  # noqa: E402
from mltgnt.memory._chroma import get_collection, query_similar, upsert_entry  # noqa: E402
from mltgnt.memory.api import append_memory_entry, memory_file_path  # noqa: E402


@pytest.fixture
def chroma_collection(tmp_path: Path):
    collection = get_collection(tmp_path, "test")
    assert collection is not None
    upsert_entry(collection, "cat1", "My cat is an orange tabby")
    upsert_entry(collection, "weather", "The weather is nice today")
    upsert_entry(collection, "neko", "Cats are cute animals")
    upsert_entry(collection, "nyanko", "A kitten is sleeping")
    return collection


def test_query_similar_semantic_synonyms(chroma_collection) -> None:
    """Entries with Japanese cat synonyms should hit a cat-related query."""
    results = query_similar(chroma_collection, "a story about cats", n_results=3)
    texts = [text for text, _score in results]

    assert len(results) >= 2
    cat_related = [t for t in texts if any(k in t for k in ("cat", "cat", "kitten"))]
    assert len(cat_related) >= 2
    assert all("weather" not in t for t in cat_related)


def test_append_memory_entry_syncs_to_chroma(tmp_path: Path) -> None:
    """After append_memory_entry, the new entry is searchable via query_similar."""
    config = MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=tmp_path / "memory",
    )
    content = "A newly added cat story"

    ok = append_memory_entry(
        config,
        "persona",
        "user",
        content,
        "2026-06-14T00:00:00+09:00",
        source_tag="file",
        under_lock=True,
    )
    assert ok is True
    assert memory_file_path(config, "persona").exists()

    collection = get_collection(config.chat_memory_dir, "persona")
    assert collection is not None

    results = query_similar(collection, "a story about cats", n_results=3)
    texts = [text for text, _score in results]
    assert any(content in t for t in texts)
