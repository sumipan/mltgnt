"""
tests/test_scoring.py — mltgnt.memory._scoring のユニットテスト

cosine_similarity の単体テスト、score_entries のソート検証
"""
from __future__ import annotations

import math

import pytest

from mltgnt.memory._scoring import ScoredEntry, cosine_similarity, score_entries


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors() -> None:
    """同一ベクトルの cosine similarity は 1.0。"""
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    """直交ベクトルの cosine similarity は 0.0。"""
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors() -> None:
    """逆方向ベクトルの cosine similarity は -1.0。"""
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_a() -> None:
    """ゼロベクトル（a）の cosine similarity は 0.0。"""
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_b() -> None:
    """ゼロベクトル（b）の cosine similarity は 0.0。"""
    assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == pytest.approx(0.0)


def test_cosine_similarity_both_zero_vectors() -> None:
    """両方ゼロベクトルの cosine similarity は 0.0。"""
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)


def test_cosine_similarity_normalized_vectors() -> None:
    """正規化済みベクトルの内積と一致する。"""
    a = [3.0, 4.0]
    b = [4.0, 3.0]
    # dot = 12 + 12 = 24, |a| = 5, |b| = 5
    expected = 24.0 / 25.0
    assert cosine_similarity(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# score_entries
# ---------------------------------------------------------------------------


def test_score_entries_returns_descending_order() -> None:
    """score_entries がスコア降順でソートされたリストを返す。"""
    query = [1.0, 0.0, 0.0]
    entries = ["entry-A", "entry-B", "entry-C"]
    entry_embeddings = [
        [0.5, 0.5, 0.0],   # 中程度の類似度
        [1.0, 0.0, 0.0],   # 最高の類似度（クエリと同じ方向）
        [0.0, 0.0, 1.0],   # 最低の類似度（直交）
    ]
    result = score_entries(query, entry_embeddings, entries)

    assert isinstance(result[0], ScoredEntry)
    assert result[0].text == "entry-B"
    assert result[1].text == "entry-A"
    assert result[2].text == "entry-C"
    assert result[0].score >= result[1].score >= result[2].score


def test_score_entries_empty_inputs() -> None:
    """空リスト入力は空リストを返す。"""
    result = score_entries([1.0, 0.0], [], [])
    assert result == []


def test_score_entries_single_entry() -> None:
    """1 エントリは 1 件のリストを返す。"""
    query = [1.0, 0.0]
    result = score_entries(query, [[1.0, 0.0]], ["single"])
    assert len(result) == 1
    assert result[0].text == "single"
    assert result[0].score == pytest.approx(1.0)


def test_scored_entry_is_frozen() -> None:
    """ScoredEntry は frozen dataclass（イミュータブル）。"""
    entry = ScoredEntry(text="test", score=0.9)
    with pytest.raises(Exception):
        entry.score = 0.5  # type: ignore[misc]
