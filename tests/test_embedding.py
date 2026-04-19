"""
tests/test_embedding.py — mltgnt.memory._embedding のユニットテスト

TC5: キャッシュヒット（同一テキストは API 呼び出しをスキップ）
TC6: キャッシュミス（テキストが変わると API が再度呼び出される）
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mltgnt.memory._embedding import EmbeddingCache, get_embeddings


def _make_call(vectors: list[list[float]]):
    """固定ベクトルを返す mock embedding_call。呼び出し回数を記録する。"""
    calls: list[list[str]] = []

    def call(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return vectors[: len(texts)]

    call.calls = calls  # type: ignore[attr-defined]
    return call


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    """キャッシュに存在しないテキストは None を返す。"""
    cache = EmbeddingCache(tmp_path / "cache")
    assert cache.get("未登録テキスト") is None


def test_cache_put_and_get(tmp_path: Path) -> None:
    """put した embedding が get で取得できる。"""
    cache = EmbeddingCache(tmp_path / "cache")
    vec = [0.1, 0.2, 0.3]
    cache.put("hello", vec)
    result = cache.get("hello")
    assert result == pytest.approx(vec)


def test_cache_different_text_miss(tmp_path: Path) -> None:
    """異なるテキストはキャッシュミスになる。"""
    cache = EmbeddingCache(tmp_path / "cache")
    cache.put("text-a", [1.0, 0.0])
    assert cache.get("text-b") is None


# ---------------------------------------------------------------------------
# TC5: キャッシュヒット — 2 回目は API 呼び出しが発生しない
# ---------------------------------------------------------------------------


def test_tc5_cache_hit_no_second_api_call(tmp_path: Path) -> None:
    """TC5: 同一テキストの embedding を 2 回取得したとき、2 回目は API 呼び出しが発生しない。"""
    cache = EmbeddingCache(tmp_path / "cache")
    call = _make_call([[0.1, 0.2, 0.3]])

    # 1 回目: API 呼び出しが発生する
    result1 = get_embeddings(["hello"], cache=cache, embedding_call=call)
    assert len(call.calls) == 1

    # 2 回目: キャッシュヒットするので API 呼び出しは発生しない
    result2 = get_embeddings(["hello"], cache=cache, embedding_call=call)
    assert len(call.calls) == 1  # まだ 1 回のまま

    assert result1 == result2


# ---------------------------------------------------------------------------
# TC6: キャッシュミス — テキストが変わると API が呼び出される
# ---------------------------------------------------------------------------


def test_tc6_cache_miss_new_api_call(tmp_path: Path) -> None:
    """TC6: テキストが変更された場合、新しい embedding が API から取得される。"""
    cache = EmbeddingCache(tmp_path / "cache")
    call = _make_call([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    get_embeddings(["text-a"], cache=cache, embedding_call=call)
    assert len(call.calls) == 1

    # 異なるテキスト → API 再呼び出し
    get_embeddings(["text-b"], cache=cache, embedding_call=call)
    assert len(call.calls) == 2


# ---------------------------------------------------------------------------
# get_embeddings: キャッシュなし
# ---------------------------------------------------------------------------


def test_get_embeddings_no_cache(tmp_path: Path) -> None:
    """cache=None の場合、毎回 API が呼び出される。"""
    call = _make_call([[0.1, 0.2], [0.3, 0.4]])

    get_embeddings(["a", "b"], cache=None, embedding_call=call)
    assert len(call.calls) == 1
    assert call.calls[0] == ["a", "b"]


def test_get_embeddings_partial_cache(tmp_path: Path) -> None:
    """一部キャッシュヒット: キャッシュミスのテキストのみ API に送られる。"""
    cache = EmbeddingCache(tmp_path / "cache")
    cache.put("cached", [1.0, 0.0])

    call = _make_call([[0.0, 1.0]])

    results = get_embeddings(["cached", "uncached"], cache=cache, embedding_call=call)
    assert len(call.calls) == 1
    assert call.calls[0] == ["uncached"]
    assert len(results) == 2
