"""tests/test_browse/test_fetcher.py — fetcher ユニットテスト (AC-2, AC-3, AC-7)"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestSSRFBlocking:
    """AC-3, AC-7: SSRF ブロックリストのテスト"""

    def test_loopback_ipv4_blocked(self):
        from skills.browse.fetcher import fetch

        with pytest.raises(ValueError, match="blocked"):
            fetch("http://127.0.0.1")

    def test_loopback_ipv6_blocked(self):
        from skills.browse.fetcher import fetch

        with pytest.raises(ValueError, match="blocked"):
            fetch("http://[::1]")

    def test_private_192_blocked(self):
        from skills.browse.fetcher import fetch

        with pytest.raises(ValueError, match="blocked"):
            fetch("http://192.168.1.1")

    def test_private_10_blocked(self):
        from skills.browse.fetcher import fetch

        with pytest.raises(ValueError, match="blocked"):
            fetch("http://10.0.0.1")

    def test_private_172_blocked(self):
        from skills.browse.fetcher import fetch

        with pytest.raises(ValueError, match="blocked"):
            fetch("http://172.16.0.1")

    def test_link_local_blocked(self):
        from skills.browse.fetcher import fetch

        with pytest.raises(ValueError, match="blocked"):
            fetch("http://169.254.1.1")


class TestFetchSuccess:
    """AC-1: 正常系テスト（httpx モック）"""

    def test_fetch_static_page_via_httpx(self):
        from skills.browse.fetcher import fetch

        html_body = "<html><body><p>Hello world</p></body></html>"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.text = html_body

        # Playwright 未インストール扱いにして httpx の結果がそのまま返ることを確認
        with patch("skills.browse.fetcher._is_private_address", return_value=False), \
             patch("skills.browse.fetcher._httpx_get", return_value=mock_response), \
             patch("skills.browse.fetcher._playwright_get", side_effect=ImportError("playwright not installed")):
            result = fetch("https://example.com")
            assert result == html_body

    def test_fetch_non_html_content_type_raises(self):
        from skills.browse.fetcher import fetch

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.text = "%PDF binary content"

        with patch("skills.browse.fetcher._is_private_address", return_value=False), \
             patch("skills.browse.fetcher._httpx_get", return_value=mock_response):
            with pytest.raises(RuntimeError, match="content.type"):
                fetch("https://example.com/file.pdf")


class TestPlaywrightFallback:
    """AC-2: Playwright フォールバックテスト"""

    def test_playwright_fallback_when_short_content(self):
        """httpx の抽出結果が短い場合に Playwright を呼び出す。"""
        from skills.browse.fetcher import fetch

        short_html = "<html><body><noscript>JS required</noscript></body></html>"
        playwright_html = "<html><body><p>" + "A" * 300 + "</p></body></html>"

        httpx_resp = MagicMock()
        httpx_resp.status_code = 200
        httpx_resp.headers = {"content-type": "text/html"}
        httpx_resp.text = short_html

        with patch("skills.browse.fetcher._is_private_address", return_value=False), \
             patch("skills.browse.fetcher._httpx_get", return_value=httpx_resp), \
             patch("skills.browse.fetcher._playwright_get", return_value=playwright_html) as mock_pw:
            result = fetch("https://spa.example.com")
            mock_pw.assert_called_once()
            assert result == playwright_html

    def test_playwright_not_called_for_rich_content(self):
        """httpx の抽出結果が十分長い場合は Playwright を呼ばない。"""
        from skills.browse.fetcher import fetch
        from skills.browse import extractor

        long_html = "<html><body><p>" + "B" * 500 + "</p></body></html>"

        httpx_resp = MagicMock()
        httpx_resp.status_code = 200
        httpx_resp.headers = {"content-type": "text/html"}
        httpx_resp.text = long_html

        with patch("skills.browse.fetcher._is_private_address", return_value=False), \
             patch("skills.browse.fetcher._httpx_get", return_value=httpx_resp), \
             patch("skills.browse.fetcher._playwright_get") as mock_pw, \
             patch.object(extractor, "extract", return_value="B" * 500):
            fetch("https://example.com")
            mock_pw.assert_not_called()

    def test_playwright_unavailable_returns_httpx_result(self):
        """Playwright 未インストール時は httpx 結果をそのまま返す。"""
        from skills.browse.fetcher import fetch

        short_html = "<html><body><noscript>JS required</noscript></body></html>"

        httpx_resp = MagicMock()
        httpx_resp.status_code = 200
        httpx_resp.headers = {"content-type": "text/html"}
        httpx_resp.text = short_html

        with patch("skills.browse.fetcher._is_private_address", return_value=False), \
             patch("skills.browse.fetcher._httpx_get", return_value=httpx_resp), \
             patch("skills.browse.fetcher._playwright_get", side_effect=ImportError("playwright not installed")):
            result = fetch("https://spa.example.com")
            assert result == short_html
