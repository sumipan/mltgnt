"""cosine similarity によるスコアリング。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
) -> list[ScoredEntry]:
    """各エントリを TF-IDF + cosine similarity でスコアリングし、スコア降順でソートして返す。

    内部で `_tfidf.vectorize()` と `cosine_similarity_matrix()` を呼び出す。

    Args:
        query: ユーザーの入力テキスト
        entries: memory エントリ本文のリスト

    Returns:
        ScoredEntry のリスト（スコア降順）
    """
    from mltgnt.memory._tfidf import vectorize

    query_vec, entry_vecs = vectorize(query, entries)
    scores = cosine_similarity_matrix(query_vec, entry_vecs)
    scored = [ScoredEntry(text=entry, score=float(score)) for entry, score in zip(entries, scores)]
    return sorted(scored, key=lambda x: x.score, reverse=True)
