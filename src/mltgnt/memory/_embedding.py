"""embedding 生成とキャッシュ。"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path

_log = logging.getLogger(__name__)

EmbeddingCall = Callable[[list[str]], list[list[float]]]
"""テキストのリストを受け取り、embedding ベクトルのリストを返す callable。"""


class EmbeddingCallError(Exception):
    """Embedding API 呼び出しエラー。"""


def default_embedding_call(texts: list[str]) -> list[list[float]]:
    """OpenAI text-embedding-3-small を呼び出す。

    環境変数 OPENAI_API_KEY を使用。
    raises: EmbeddingCallError（API エラー時）
    """
    try:
        import openai
    except ImportError as e:
        raise EmbeddingCallError(
            "openai パッケージが未インストール: pip install mltgnt[embedding]"
        ) from e

    try:
        client = openai.OpenAI()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        raise EmbeddingCallError(f"OpenAI embedding API エラー: {e}") from e


class EmbeddingCache:
    """ファイルベースの embedding キャッシュ。

    Args:
        cache_dir: キャッシュディレクトリパス
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def get(self, text: str) -> list[float] | None:
        """キャッシュからテキストの embedding を取得。ミスなら None。"""
        path = self._path(self._key(text))
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, text: str, embedding: list[float]) -> None:
        """キャッシュに embedding を保存。"""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(self._key(text))
        try:
            path.write_text(json.dumps(embedding), encoding="utf-8")
        except OSError as e:
            _log.warning("EmbeddingCache: キャッシュ書き込み失敗: %s", e)


def get_embeddings(
    texts: list[str],
    *,
    cache: EmbeddingCache | None = None,
    embedding_call: EmbeddingCall | None = None,
) -> list[list[float]]:
    """テキストリストの embedding を返す。キャッシュがあればキャッシュを優先。

    Args:
        texts: embedding を取得するテキストのリスト
        cache: EmbeddingCache インスタンス（None ならキャッシュなし）
        embedding_call: embedding API 呼び出し（None なら default_embedding_call）

    Returns:
        texts と同じ順序の embedding ベクトルリスト
    """
    call = embedding_call or default_embedding_call

    if not texts:
        return []

    if cache is None:
        return call(texts)

    # キャッシュヒット/ミスを分離して処理
    results: list[list[float] | None] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = cache.get(text)
        if cached is not None:
            results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        new_embeddings = call(uncached_texts)
        for idx, embedding in zip(uncached_indices, new_embeddings):
            cache.put(texts[idx], embedding)
            results[idx] = embedding

    return results  # type: ignore[return-value]
