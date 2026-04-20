"""
tests/test_tfidf.py — mltgnt.memory._tfidf のユニットテスト

TC5（日本語テキスト対応）を含む
"""
from __future__ import annotations

from mltgnt.memory._tfidf import vectorize


# ---------------------------------------------------------------------------
# vectorize: 基本的な形状検証
# ---------------------------------------------------------------------------


def test_vectorize_returns_two_arrays() -> None:
    """vectorize は (query_vec, entry_vecs) のタプルを返す。"""
    query_vec, entry_vecs = vectorize("hello world", ["foo bar", "baz qux"])
    assert query_vec.shape[0] == 1
    assert entry_vecs.shape[0] == 2
    assert query_vec.shape[1] == entry_vecs.shape[1]


def test_vectorize_dimension_consistency() -> None:
    """クエリベクトルとエントリ行列の次元数が一致する。"""
    query_vec, entry_vecs = vectorize("python decorator", ["python code", "weather today", "cooking recipe"])
    assert query_vec.shape == (1, entry_vecs.shape[1])
    assert entry_vecs.shape == (3, query_vec.shape[1])


def test_vectorize_single_entry() -> None:
    """TC10: エントリが 1 件のみでも動作する（コーパスサイズ=2: クエリ+エントリ）。"""
    query_vec, entry_vecs = vectorize("hello", ["world"])
    assert query_vec.shape[0] == 1
    assert entry_vecs.shape[0] == 1
    assert query_vec.shape[1] == entry_vecs.shape[1]


def test_vectorize_japanese_text() -> None:
    """TC5: 日本語テキストに対して TF-IDF ベクトル化が正常に動作する。"""
    query_vec, entry_vecs = vectorize(
        "Python のデコレータについて教えて",
        [
            "料理のレシピについて話した。おいしいパスタの作り方を学んだ。",
            "Python のデコレータについて調べた。コードの再利用性が高まる。",
            "今日の天気は晴れだった。気温が上がってきた。",
        ],
    )
    assert query_vec.shape[0] == 1
    assert entry_vecs.shape[0] == 3
    # ベクトルが全ゼロでないこと（少なくとも1要素が非ゼロ）
    assert (query_vec != 0).any() or (entry_vecs != 0).any()


def test_vectorize_returns_nonnegative_values() -> None:
    """TF-IDF は非負値のみを返す。"""
    query_vec, entry_vecs = vectorize("test query", ["entry one", "entry two"])
    assert (query_vec >= 0).all()
    assert (entry_vecs >= 0).all()


def test_vectorize_overlapping_terms() -> None:
    """クエリとエントリに共通語がある場合、関連エントリのベクトルが非ゼロになる。"""
    query_vec, entry_vecs = vectorize("python programming", ["python code review", "cooking recipe"])
    # python を含むエントリ(0) のベクトルが非ゼロになるはず
    assert (entry_vecs[0] != 0).any()
