"""skills/browse/extractor.py — HTML → Markdown テキスト抽出 (trafilatura + BeautifulSoup フォールバック)"""
from __future__ import annotations


def extract(html: str, *, url: str = "") -> str:
    """HTML から本文を Markdown 形式で抽出する。

    Args:
        html: HTML 文字列
        url: 元 URL（trafilatura のメタデータ抽出用）

    Returns:
        Markdown 形式の本文テキスト
    """
    if not html:
        return ""

    try:
        import trafilatura

        result = trafilatura.extract(
            html,
            url=url or None,
            output_format="txt",
            include_links=True,
            include_tables=True,
            favor_recall=True,
        )
        if result:
            return result
    except Exception:
        pass

    # BeautifulSoup フォールバック
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        body = soup.find("body") or soup
        return body.get_text(separator="\n", strip=True)
    except Exception:
        return ""
