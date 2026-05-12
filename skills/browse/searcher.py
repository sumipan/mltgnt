"""skills/browse/searcher.py — DuckDuckGo 検索 → URL リスト → 本文取得"""
from __future__ import annotations

from duckduckgo_search import DDGS

from skills.browse.extractor import extract
from skills.browse.fetcher import fetch


def search(query: str, *, top: int = 3) -> list[dict]:
    """DuckDuckGo でクエリを検索し、上位 N 件の URL・タイトル・本文を返す。

    Args:
        query: 検索クエリ文字列
        top: 取得件数（1〜10、デフォルト 3）

    Returns:
        [{"url": str, "title": str, "content": str}, ...]

    Raises:
        ValueError: top が範囲外
    """
    if top < 1 or top > 10:
        raise ValueError(f"top must be between 1 and 10, got {top}")

    with DDGS() as ddgs:
        ddg_results = list(ddgs.text(query, max_results=top))[:top]

    results = []
    for item in ddg_results:
        url = item.get("href", "")
        title = item.get("title", "")
        try:
            html = fetch(url)
            content = extract(html, url=url)
        except Exception:
            continue

        results.append({"url": url, "title": title, "content": content})

    return results
