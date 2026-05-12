"""tests/test_browse/test_searcher.py — searcher ユニットテスト (AC-4, AC-7)"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestSearchTopLimit:
    """AC-4: top パラメータのバリデーション"""

    def test_top_11_raises_error(self):
        from skills.browse.searcher import search

        with pytest.raises(ValueError, match="top"):
            search("test query", top=11)

    def test_top_0_raises_error(self):
        from skills.browse.searcher import search

        with pytest.raises(ValueError, match="top"):
            search("test query", top=0)

    def test_top_10_is_allowed(self):
        from skills.browse.searcher import search

        ddg_results = [
            {"href": f"https://example.com/{i}", "title": f"Title {i}", "body": ""}
            for i in range(10)
        ]

        with patch("skills.browse.searcher.DDGS") as mock_ddgs, \
             patch("skills.browse.searcher.fetch", return_value="<html><body><p>content</p></body></html>"), \
             patch("skills.browse.searcher.extract", return_value="content text"):
            mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_ddgs.return_value)
            mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)
            mock_ddgs.return_value.text = MagicMock(return_value=ddg_results)
            results = search("test query", top=10)
            assert len(results) <= 10


class TestSearchResults:
    """AC-4: 検索結果の形式"""

    def _mock_search(self, n=3):
        from skills.browse.searcher import search

        ddg_results = [
            {"href": f"https://example.com/{i}", "title": f"Title {i}", "body": "snippet"}
            for i in range(n)
        ]

        with patch("skills.browse.searcher.DDGS") as mock_ddgs, \
             patch("skills.browse.searcher.fetch", return_value="<html><body><p>content</p></body></html>"), \
             patch("skills.browse.searcher.extract", return_value="content text"):
            mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_ddgs.return_value)
            mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)
            mock_ddgs.return_value.text = MagicMock(return_value=ddg_results)
            return search("test query", top=n)

    def test_returns_list_of_dicts(self):
        results = self._mock_search(3)
        assert isinstance(results, list)
        for r in results:
            assert "url" in r
            assert "title" in r
            assert "content" in r

    def test_default_top_3(self):
        from skills.browse.searcher import search

        ddg_results = [
            {"href": f"https://example.com/{i}", "title": f"Title {i}", "body": ""}
            for i in range(5)
        ]

        with patch("skills.browse.searcher.DDGS") as mock_ddgs, \
             patch("skills.browse.searcher.fetch", return_value="<html><body><p>content</p></body></html>"), \
             patch("skills.browse.searcher.extract", return_value="content text"):
            mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_ddgs.return_value)
            mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)
            mock_ddgs.return_value.text = MagicMock(return_value=ddg_results)
            results = search("test query")
            assert len(results) <= 3


class TestSearchFailureSkip:
    """AC-4: 個別 URL の取得失敗はスキップされる"""

    def test_failed_url_skipped(self):
        from skills.browse.searcher import search

        ddg_results = [
            {"href": "https://ok.example.com", "title": "OK", "body": ""},
            {"href": "https://fail.example.com", "title": "Fail", "body": ""},
        ]

        def mock_fetch(url, **kwargs):
            if "fail" in url:
                raise RuntimeError("Connection failed")
            return "<html><body><p>content</p></body></html>"

        with patch("skills.browse.searcher.DDGS") as mock_ddgs, \
             patch("skills.browse.searcher.fetch", side_effect=mock_fetch), \
             patch("skills.browse.searcher.extract", return_value="content text"):
            mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_ddgs.return_value)
            mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)
            mock_ddgs.return_value.text = MagicMock(return_value=ddg_results)
            results = search("test query", top=2)
            assert len(results) == 1
            assert results[0]["url"] == "https://ok.example.com"
