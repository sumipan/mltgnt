"""Chroma 不可時の TF-IDF フォールバックテスト。"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.config import MemoryConfig
from mltgnt.memory._scoring import ScoredEntry, score_entries
from mltgnt.memory.api import append_memory_entry, memory_file_path


def test_score_entries_without_chromadb_uses_tfidf() -> None:
    """chromadb 未インストール相当（collection=None）で TF-IDF が動作する。"""
    entries = [
        "Python のデコレータについて調べた。",
        "今日の天気は晴れだった。",
    ]
    scored = score_entries("Python デコレータ", entries, chroma_collection=None)

    assert len(scored) == 2
    assert all(isinstance(item, ScoredEntry) for item in scored)
    assert scored[0].text == entries[0]
    assert scored[0].score >= scored[1].score


def test_get_collection_init_failure_returns_none(tmp_path: Path, caplog) -> None:
    """Chroma 初期化失敗時に warning を出して None を返す。"""
    with patch("mltgnt.memory._chroma._import_chromadb") as mock_import:
        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.side_effect = RuntimeError("init failed")
        mock_import.return_value = mock_chromadb

        with caplog.at_level(logging.WARNING, logger="mltgnt.memory._chroma"):
            from mltgnt.memory._chroma import get_collection

            result = get_collection(tmp_path, "persona")

    assert result is None
    assert any("Chroma init failed" in r.message for r in caplog.records)


def test_score_entries_chroma_query_failure_falls_back_to_tfidf(caplog) -> None:
    """Chroma クエリ例外時に TF-IDF へフォールバックする。"""
    mock_collection = MagicMock()
    entries = ["Python のデコレータ", "天気の話"]

    with patch("mltgnt.memory._chroma.query_similar", side_effect=RuntimeError("query failed")):
        with caplog.at_level(logging.WARNING, logger="mltgnt.memory._scoring"):
            scored = score_entries("Python", entries, chroma_collection=mock_collection)

    assert len(scored) == 2
    assert any("Chroma query failed" in r.message for r in caplog.records)


def test_append_memory_entry_chroma_upsert_failure_still_writes_jsonl(
    tmp_path: Path, caplog
) -> None:
    """Chroma upsert 失敗時も JSONL 書き込みは成功し warning のみ。"""
    config = MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=tmp_path / "memory",
    )

    with patch("mltgnt.memory._chroma.get_collection") as mock_get:
        mock_collection = MagicMock()
        mock_collection.upsert.side_effect = RuntimeError("upsert failed")
        mock_get.return_value = mock_collection

        with caplog.at_level(logging.WARNING, logger="mltgnt.memory.api"):
            ok = append_memory_entry(
                config,
                "persona",
                "user",
                "テスト内容",
                "2026-06-14T00:00:00+09:00",
                source_tag="file",
                under_lock=True,
            )

    assert ok is True
    assert memory_file_path(config, "persona").exists()
    assert any("Chroma upsert failed" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "test_module",
    ["tests/test_memory_relevance.py"],
)
def test_existing_relevance_tests_pass_without_chroma(test_module: str) -> None:
    """chromadb 未使用でも既存 relevance テストが通る（import 確認）。"""
    import importlib

    mod = importlib.import_module(test_module.replace("/", ".").replace(".py", ""))
    assert mod is not None
