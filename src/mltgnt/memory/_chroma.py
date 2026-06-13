"""Chroma ベクトル DB による意味検索バックエンド。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb

_log = logging.getLogger(__name__)


def _import_chromadb() -> "type[chromadb] | None":
    try:
        import chromadb as _chromadb

        return _chromadb
    except ImportError:
        return None


def get_collection(
    memory_dir: Path, persona_stem: str
) -> "chromadb.Collection | None":
    """ペルソナ用 Chroma コレクションを取得する。不可時は None。"""
    chromadb = _import_chromadb()
    if chromadb is None:
        return None
    try:
        persist_dir = memory_dir / ".chroma" / persona_stem
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))
        return client.get_or_create_collection(name=persona_stem)
    except Exception as exc:
        _log.warning("get_collection: Chroma init failed, fallback to TF-IDF: %s", exc)
        return None


def upsert_entry(
    collection: "chromadb.Collection",
    entry_id: str,
    text: str,
) -> None:
    """エントリを Chroma コレクションに upsert する。"""
    collection.upsert(ids=[entry_id], documents=[text])


def query_similar(
    collection: "chromadb.Collection",
    query_text: str,
    *,
    n_results: int,
) -> list[tuple[str, float]]:
    """クエリに意味的に近いドキュメントを返す。

    Returns:
        (document_text, similarity_score) のリスト（スコア降順）
    """
    raw = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "distances"],
    )
    documents = raw.get("documents") or [[]]
    distances = raw.get("distances") or [[]]
    if not documents[0]:
        return []

    scored: list[tuple[str, float]] = []
    for doc, dist in zip(documents[0], distances[0]):
        if doc is None:
            continue
        # Chroma cosine distance: 0 = identical; convert to similarity in [0, 1]
        similarity = max(0.0, 1.0 - float(dist))
        scored.append((doc, similarity))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored
