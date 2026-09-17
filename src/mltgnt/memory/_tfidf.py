"""TF-IDF-based text vectorization."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize(
    query: str,
    entries: list[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert a query and memory entries to TF-IDF vectors.

    Fit/transform query + all entries as one corpus with TfidfVectorizer,
    returning a query vector (1 row) and entry matrix (N rows).

    Args:
        query: User input text
        entries: List of memory entry bodies

    Returns:
        (query_vec, entry_vecs):
            query_vec — TF-IDF vector of shape (1, D)
            entry_vecs — TF-IDF matrix of shape (N, D)
    """
    corpus = [query] + entries
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    matrix = vectorizer.fit_transform(corpus).toarray()
    query_vec: NDArray[np.float64] = matrix[:1]
    entry_vecs: NDArray[np.float64] = matrix[1:]
    return query_vec, entry_vecs
