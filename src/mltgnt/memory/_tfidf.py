"""TF-IDF ベースのテキストベクトル化。"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize(
    query: str,
    entries: list[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """クエリと memory エントリ群を TF-IDF ベクトルに変換する。

    TfidfVectorizer でクエリ + 全エントリを同一コーパスとして学習・変換し、
    クエリベクトル（1行）とエントリ行列（N行）を返す。

    Args:
        query: ユーザーの入力テキスト
        entries: memory エントリ本文のリスト

    Returns:
        (query_vec, entry_vecs):
            query_vec — shape (1, D) の TF-IDF ベクトル
            entry_vecs — shape (N, D) の TF-IDF ベクトル行列
    """
    corpus = [query] + entries
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    matrix = vectorizer.fit_transform(corpus).toarray()
    query_vec: NDArray[np.float64] = matrix[:1]
    entry_vecs: NDArray[np.float64] = matrix[1:]
    return query_vec, entry_vecs
