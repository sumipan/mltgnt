"""skills/browse/fetcher.py — URL 取得 (httpx → Playwright フォールバック)"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def _is_private_address(url: str) -> bool:
    """URL のホストが RFC 1918 / ループバック / リンクローカルか判定する。"""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return False

    # IPv6 ブラケット除去は urlparse が処理済み
    try:
        addr = ipaddress.ip_address(hostname)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        pass

    # ホスト名を DNS 解決してチェック
    try:
        infos = socket.getaddrinfo(hostname, None)
        for info in infos:
            ip_str = info[4][0]
            # IPv6 スコープ ID を除去
            ip_str = ip_str.split("%")[0]
            addr = ipaddress.ip_address(ip_str)
            if addr.is_private or addr.is_loopback or addr.is_link_local:
                return True
    except (socket.gaierror, ValueError):
        pass

    return False


def _httpx_get(url: str, timeout: int = 10):
    import httpx

    return httpx.get(url, follow_redirects=True, max_redirects=3, timeout=timeout)


def _playwright_get(url: str, timeout: int = 30) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
        content = page.content()
        context.close()
        browser.close()
        return content


def fetch(url: str, *, timeout_light: int = 10, timeout_heavy: int = 30) -> str:
    """URL から HTML を取得して返す。

    Args:
        url: 取得対象 URL
        timeout_light: httpx タイムアウト秒数
        timeout_heavy: Playwright タイムアウト秒数

    Returns:
        HTML 文字列

    Raises:
        ValueError: ブロックリスト該当 URL
        TimeoutError: 両経路ともタイムアウト
        RuntimeError: 取得失敗
    """
    if _is_private_address(url):
        raise ValueError(f"blocked private/loopback address: {url}")

    try:
        response = _httpx_get(url, timeout=timeout_light)
    except Exception as e:
        import httpx

        if isinstance(e, httpx.TimeoutException):
            raise TimeoutError(f"httpx timeout: {url}") from e
        raise RuntimeError(f"httpx fetch failed: {e}") from e

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        raise RuntimeError(f"content-type is not text/html: {content_type!r} for {url}")

    html = response.text

    # Playwright フォールバック判定
    needs_playwright = "<noscript>" in html
    if not needs_playwright:
        from skills.browse.extractor import extract as _extract

        extracted = _extract(html, url=url)
        needs_playwright = len(extracted) < 200

    if needs_playwright:
        try:
            html = _playwright_get(url, timeout=timeout_heavy)
        except ImportError:
            pass  # Playwright 未インストール時は httpx 結果をそのまま使う
        except Exception as e:
            pass  # Playwright 失敗時も httpx 結果を返す

    return html
