"""
tests/test_tfidf.py — unit tests for mltgnt.memory._tfidf

Includes TC5 (Japanese text support).
"""
from __future__ import annotations

from mltgnt.memory._tfidf import vectorize


# ---------------------------------------------------------------------------
# vectorize: basic shape checks
# ---------------------------------------------------------------------------


def test_vectorize_returns_two_arrays() -> None:
    """vectorize returns a (query_vec, entry_vecs) tuple."""
    query_vec, entry_vecs = vectorize("hello world", ["foo bar", "baz qux"])
    assert query_vec.shape[0] == 1
    assert entry_vecs.shape[0] == 2
    assert query_vec.shape[1] == entry_vecs.shape[1]


def test_vectorize_dimension_consistency() -> None:
    """Query vector and entry matrix share the same feature dimension."""
    query_vec, entry_vecs = vectorize("python decorator", ["python code", "weather today", "cooking recipe"])
    assert query_vec.shape == (1, entry_vecs.shape[1])
    assert entry_vecs.shape == (3, query_vec.shape[1])


def test_vectorize_single_entry() -> None:
    """TC10: works with a single entry (corpus size=2: query + entry)."""
    query_vec, entry_vecs = vectorize("hello", ["world"])
    assert query_vec.shape[0] == 1
    assert entry_vecs.shape[0] == 1
    assert query_vec.shape[1] == entry_vecs.shape[1]


def test_vectorize_japanese_text() -> None:
    """TC5: TF-IDF vectorization works on Japanese text."""
    query_vec, entry_vecs = vectorize(
        "Python  explain decorators",
        [
            "Discussed recipes and learned to make pasta.",
            "Python  decorators were studied. Code reuse improved.",
            "The weather was sunny and warmer today.",
        ],
    )
    assert query_vec.shape[0] == 1
    assert entry_vecs.shape[0] == 3
    # Vectors must not be all-zero (at least one non-zero element)
    assert (query_vec != 0).any() or (entry_vecs != 0).any()


def test_vectorize_returns_nonnegative_values() -> None:
    """TF-IDF returns only non-negative values."""
    query_vec, entry_vecs = vectorize("test query", ["entry one", "entry two"])
    assert (query_vec >= 0).all()
    assert (entry_vecs >= 0).all()


def test_vectorize_overlapping_terms() -> None:
    """When query and entry share terms, the related entry vector is non-zero."""
    query_vec, entry_vecs = vectorize("python programming", ["python code review", "cooking recipe"])
    # Entry 0 contains "python" and should be non-zero
    assert (entry_vecs[0] != 0).any()
