"""cosine similarity によるスコアリング。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import chromadb

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredEntry:
    """スコア付き memory エントリ。"""

    text: str    # エントリ本文
    score: float  # cosine similarity（0.0 〜 1.0、TF-IDF は非負）


def cosine_similarity_matrix(
    query_vec: NDArray[np.float64],
    entry_vecs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """クエリベクトルと各エントリベクトルの cosine similarity を返す。

    Args:
        query_vec: shape (1, D)
        entry_vecs: shape (N, D)

    Returns:
        shape (N,) の similarity スコア配列
    """
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0.0:
        return np.zeros(entry_vecs.shape[0])

    entry_norms = np.linalg.norm(entry_vecs, axis=1)
    safe_entry_norms = np.where(entry_norms == 0.0, 1.0, entry_norms)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        dots = (entry_vecs @ query_vec.T)[:, 0]  # shape (N,)
    scores = np.nan_to_num(dots / safe_entry_norms / query_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return scores


def score_entries(
    query: str,
    entries: list[str],
    *,
    chroma_collection: "chromadb.Collection | None" = None,
) -> list[ScoredEntry]:
    """各エントリをスコアリングし、スコア降順でソートして返す。

    Chroma コレクションが利用可能な場合は意味検索を優先する。
    不可時または失敗時は TF-IDF + cosine similarity にフォールバックする。

    Args:
        query: ユーザーの入力テキスト
        entries: memory エントリ本文のリスト
        chroma_collection: Chroma コレクション（None なら TF-IDF）

    Returns:
        ScoredEntry のリスト（スコア降順）
    """
    if chroma_collection is not None:
        try:
            from mltgnt.memory._chroma import query_similar

            chroma_results = query_similar(
                chroma_collection, query, n_results=len(entries)
            )
            if chroma_results:
                entry_set = set(entries)
                scored: list[ScoredEntry] = []
                seen: set[str] = set()
                for text, score in chroma_results:
                    if text in entry_set and text not in seen:
                        scored.append(ScoredEntry(text=text, score=score))
                        seen.add(text)
                if scored:
                    return sorted(scored, key=lambda x: x.score, reverse=True)
        except Exception as exc:
            _log.warning(
                "score_entries: Chroma query failed, fallback to TF-IDF: %s", exc
            )

    from mltgnt.memory._tfidf import vectorize

    query_vec, entry_vecs = vectorize(query, entries)
    scores = cosine_similarity_matrix(query_vec, entry_vecs)
    scored = [ScoredEntry(text=entry, score=float(score)) for entry, score in zip(entries, scores)]
    return sorted(scored, key=lambda x: x.score, reverse=True)
