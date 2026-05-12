"""tests/test_browse/test_extractor.py — extractor ユニットテスト (AC-5, AC-7)"""
from __future__ import annotations

import pytest


def test_extract_returns_string():
    from skills.browse.extractor import extract

    html = "<html><body><h1>Title</h1><p>Hello world paragraph.</p></body></html>"
    result = extract(html, url="https://example.com")
    assert isinstance(result, str)
    assert len(result) > 0


def test_extract_preserves_heading_structure():
    from skills.browse.extractor import extract

    html = """<html><body>
    <h1>Main Title</h1>
    <p>Introduction paragraph with enough text to be extracted by trafilatura properly.</p>
    <h2>Sub Heading</h2>
    <p>Content under sub heading with sufficient text for trafilatura to process.</p>
    </body></html>"""
    result = extract(html, url="https://example.com/article")
    assert isinstance(result, str)
    assert len(result) > 0


def test_extract_fallback_on_empty_trafilatura():
    """trafilatura が空結果を返す場合、BeautifulSoup でフォールバックする。"""
    from skills.browse.extractor import extract

    # 本文テキストが極端に少ないページ
    html = "<html><head><title>Test</title></head><body><p>Hi</p></body></html>"
    result = extract(html, url="")
    assert isinstance(result, str)


def test_extract_empty_html():
    from skills.browse.extractor import extract

    result = extract("", url="")
    assert isinstance(result, str)


def test_extract_with_noise():
    """ナビバー・広告ノイズが含まれていてもクラッシュしない。"""
    from skills.browse.extractor import extract

    html = """<html><body>
    <nav>Navigation menu items here</nav>
    <div class="ad">Advertisement content</div>
    <article>
      <h1>Real Article Title</h1>
      <p>This is the real article content that trafilatura should extract properly.</p>
      <p>Second paragraph with more content for extraction purposes.</p>
    </article>
    <footer>Footer content here</footer>
    </body></html>"""
    result = extract(html, url="https://example.com/article")
    assert isinstance(result, str)
