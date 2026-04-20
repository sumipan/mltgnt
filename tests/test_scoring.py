"""
tests/test_scoring.py — mltgnt.memory._scoring のユニットテスト

cosine_similarity_matrix の単体テスト、score_entries のソート検証

Note: Issue #198 で embedding ベースから TF-IDF ベースに変更。
旧 cosine_similarity(list, list) / score_entries(query_emb, entry_embs, entries) は
TF-IDF ベースの新インタフェースに置き換えた。
"""
from __future__ import annotations

import numpy as np
import pytest

from mltgnt.memory._scoring import ScoredEntry, cosine_similarity_matrix, score_entries


# ---------------------------------------------------------------------------
# cosine_similarity_matrix
# ---------------------------------------------------------------------------


def test_cosine_similarity_matrix_identical_vector() -> None:
    """クエリと同一エントリの cosine similarity は 1.0。"""
    query_vec = np.array([[1.0, 0.0, 0.0]])
    entry_vecs = np.array([[1.0, 0.0, 0.0]])
    result = cosine_similarity_matrix(query_vec, entry_vecs)
    assert result[0] == pytest.approx(1.0)


def test_cosine_similarity_matrix_orthogonal_vectors() -> None:
    """直交ベクトルの cosine similarity は 0.0。"""
    query_vec = np.array([[1.0, 0.0]])
    entry_vecs = np.array([[0.0, 1.0]])
    result = cosine_similarity_matrix(query_vec, entry_vecs)
    assert result[0] == pytest.approx(0.0)


def test_cosine_similarity_matrix_multiple_entries() -> None:
    """複数エントリに対して shape (N,) の配列を返す。"""
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
    """クエリがゼロベクトルの場合は 0.0 を返す。"""
    query_vec = np.array([[0.0, 0.0]])
    entry_vecs = np.array([[1.0, 2.0]])
    result = cosine_similarity_matrix(query_vec, entry_vecs)
    assert result[0] == pytest.approx(0.0)


def test_cosine_similarity_matrix_nonnegative_for_tfidf() -> None:
    """TF-IDF ベクトル（非負値）に対する cosine similarity は 0.0 以上。"""
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
    """score_entries がスコア降順でソートされたリストを返す。"""
    entries = [
        "python decorator code programming",  # プログラミング
        "cooking recipe pasta delicious",     # 料理
        "weather sunny temperature today",    # 天気
    ]
    result = score_entries("python programming decorator", entries)

    assert isinstance(result[0], ScoredEntry)
    assert result[0].score >= result[1].score >= result[2].score
    # プログラミングエントリが最上位
    assert result[0].text == entries[0]


def test_score_entries_returns_all_entries() -> None:
    """全エントリがスコア付きで返る。"""
    entries = ["entry-A text foo", "entry-B text bar", "entry-C text baz"]
    result = score_entries("query text", entries)
    assert len(result) == 3
    texts = {s.text for s in result}
    assert texts == set(entries)


def test_score_entries_single_entry() -> None:
    """TC10: 1 エントリでも正常に動作する。"""
    result = score_entries("hello world", ["hello world example"])
    assert len(result) == 1
    assert isinstance(result[0], ScoredEntry)
    assert result[0].score >= 0.0


def test_score_entries_score_range() -> None:
    """TF-IDF ベースのスコアは 0.0 以上 1.0 以下。"""
    entries = ["foo bar", "baz qux", "hello world"]
    result = score_entries("foo", entries)
    for s in result:
        assert 0.0 <= s.score <= 1.0 + 1e-9  # float 誤差を許容


def test_scored_entry_is_frozen() -> None:
    """ScoredEntry は frozen dataclass（イミュータブル）。"""
    entry = ScoredEntry(text="test", score=0.9)
    with pytest.raises(Exception):
        entry.score = 0.5  # type: ignore[misc]
