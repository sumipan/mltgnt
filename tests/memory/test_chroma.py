"""Chroma 意味検索バックエンドの単体テスト。"""
from __future__ import annotations

from pathlib import Path

import pytest

chromadb = pytest.importorskip("chromadb")

from mltgnt.config import MemoryConfig
from mltgnt.memory._chroma import get_collection, query_similar, upsert_entry
from mltgnt.memory.api import append_memory_entry, memory_file_path


@pytest.fixture
def chroma_collection(tmp_path: Path):
    collection = get_collection(tmp_path, "test")
    assert collection is not None
    upsert_entry(collection, "cat1", "うちの猫は茶トラです")
    upsert_entry(collection, "weather", "今日は天気がいい")
    upsert_entry(collection, "neko", "ネコはかわいい動物です")
    upsert_entry(collection, "nyanko", "にゃんこが寝ている")
    return collection


def test_query_similar_semantic_synonyms(chroma_collection) -> None:
    """同義語（ネコ・にゃんこ）を含むエントリが「猫の話」クエリでヒットする。"""
    results = query_similar(chroma_collection, "猫の話", n_results=3)
    texts = [text for text, _score in results]

    assert len(results) >= 2
    cat_related = [t for t in texts if any(k in t for k in ("猫", "ネコ", "にゃんこ"))]
    assert len(cat_related) >= 2
    assert all("天気" not in t for t in cat_related)


def test_append_memory_entry_syncs_to_chroma(tmp_path: Path) -> None:
    """append_memory_entry 後、query_similar で追加エントリが検索可能。"""
    config = MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=tmp_path / "memory",
    )
    content = "新しく追加した猫のエピソード"

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

    results = query_similar(collection, "猫の話", n_results=3)
    texts = [text for text, _score in results]
    assert any(content in t for t in texts)
