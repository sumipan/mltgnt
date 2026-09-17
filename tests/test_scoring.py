"""
tests/test_scoring.py — unit tests for mltgnt.memory._scoring

Unit tests for cosine_similarity_matrix and sort verification for score_entries.

Note: Issue #198 switched from embedding-based to TF-IDF-based scoring.
The old cosine_similarity(list, list) / score_entries(query_emb, entry_embs, entries)
interfaces were replaced with the TF-IDF-based API.
"""
from __future__ import annotations

import numpy as np
import pytest

from mltgnt.memory._scoring import ScoredEntry, cosine_similarity_matrix, score_entries


# ---------------------------------------------------------------------------
# cosine_similarity_matrix
# ---------------------------------------------------------------------------


def test_cosine_similarity_matrix_identical_vector() -> None:
    """Cosine similarity of a query with an identical entry is 1.0."""
    query_vec = np.array([[1.0, 0.0, 0.0]])
    entry_vecs = np.array([[1.0, 0.0, 0.0]])
    result = cosine_similarity_matrix(query_vec, entry_vecs)
    assert result[0] == pytest.approx(1.0)


def test_cosine_similarity_matrix_orthogonal_vectors() -> None:
    """Cosine similarity of orthogonal vectors is 0.0."""
    query_vec = np.array([[1.0, 0.0]])
    entry_vecs = np.array([[0.0, 1.0]])
    result = cosine_similarity_matrix(query_vec, entry_vecs)
    assert result[0] == pytest.approx(0.0)


def test_cosine_similarity_matrix_multiple_entries() -> None:
    """Returns a shape-(N,) array for multiple entries."""
    query_vec = np.array([[1.0, 0.0, 0.0]])
    entry_vecs = np.array([
        [1.0, 0.0, 0.0],  # sim=1.0
        [0.0, 1.0, 0.0],  # sim=0.0
        [0.0, 0.0, 1.0],  # sim=0.0
    ])
    result = cosine_similarity_matrix(query_vec, entry_vecs)
    assert result.shape == (3,)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(0.0)


def test_cosine_similarity_matrix_zero_query_vec() -> None:
    """A zero query vector yields 0.0."""
    query_vec = np.array([[0.0, 0.0]])
    entry_vecs = np.array([[1.0, 2.0]])
    result = cosine_similarity_matrix(query_vec, entry_vecs)
    assert result[0] == pytest.approx(0.0)


def test_cosine_similarity_matrix_nonnegative_for_tfidf() -> None:
    """Cosine similarity over non-negative TF-IDF vectors is >= 0.0."""
    query_vec = np.array([[0.5, 0.3, 0.0]])
    entry_vecs = np.array([
        [0.4, 0.0, 0.6],
        [0.0, 0.8, 0.2],
    ])
    result = cosine_similarity_matrix(query_vec, entry_vecs)
    assert (result >= 0).all()


# ---------------------------------------------------------------------------
# score_entries
# ---------------------------------------------------------------------------


def test_score_entries_returns_descending_order() -> None:
    """score_entries returns a list sorted by score descending."""
    entries = [
        "python decorator code programming",  # programming
        "cooking recipe pasta delicious",     # cooking
        "weather sunny temperature today",    # weather
    ]
    result = score_entries("python programming decorator", entries)

    assert isinstance(result[0], ScoredEntry)
    assert result[0].score >= result[1].score >= result[2].score
    # Programming entry should rank first
    assert result[0].text == entries[0]


def test_score_entries_returns_all_entries() -> None:
    """All entries are returned with scores."""
    entries = ["entry-A text foo", "entry-B text bar", "entry-C text baz"]
    result = score_entries("query text", entries)
    assert len(result) == 3
    texts = {s.text for s in result}
    assert texts == set(entries)


def test_score_entries_single_entry() -> None:
    """TC10: works with a single entry."""
    result = score_entries("hello world", ["hello world example"])
    assert len(result) == 1
    assert isinstance(result[0], ScoredEntry)
    assert result[0].score >= 0.0


def test_score_entries_score_range() -> None:
    """TF-IDF-based scores are in [0.0, 1.0]."""
    entries = ["foo bar", "baz qux", "hello world"]
    result = score_entries("foo", entries)
    for s in result:
        assert 0.0 <= s.score <= 1.0 + 1e-9  # allow float error


def test_scored_entry_is_frozen() -> None:
    """ScoredEntry is a frozen (immutable) dataclass."""
    entry = ScoredEntry(text="test", score=0.9)
    with pytest.raises(Exception):
        entry.score = 0.5  # type: ignore[misc]
