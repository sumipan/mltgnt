"""cosine similarity によるスコアリング。"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredEntry:
    """スコア付き memory エントリ。"""

    text: str    # エントリ本文
    score: float  # cosine similarity（-1.0 〜 1.0）


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """2 ベクトルの cosine similarity を返す。ゼロベクトルの場合は 0.0。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_entries(
    query_embedding: list[float],
    entry_embeddings: list[list[float]],
    entries: list[str],
) -> list[ScoredEntry]:
    """各エントリをスコアリングし、スコア降順でソートして返す。

    Args:
        query_embedding: クエリの embedding ベクトル
        entry_embeddings: 各エントリの embedding ベクトル（entries と同順）
        entries: memory エントリ本文のリスト

    Returns:
        ScoredEntry のリスト（スコア降順）
    """
    scored = [
        ScoredEntry(text=entry, score=cosine_similarity(query_embedding, emb))
        for entry, emb in zip(entries, entry_embeddings)
    ]
    return sorted(scored, key=lambda x: x.score, reverse=True)
