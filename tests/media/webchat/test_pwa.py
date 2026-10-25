"""PWA manifest, service worker, icons, and UI contracts for WebChat."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mltgnt.media.webchat.app import create_app  # noqa: E402
from mltgnt.media.webchat.config import WebChatMediaConfig  # noqa: E402
from mltgnt.media.webchat.pwa import manifest, resolve_pwa_icon, service_worker_source  # noqa: E402
from mltgnt.media.webchat.ui import INDEX_HTML  # noqa: E402


class _Bridge:
    def handle_event(self, _event: object) -> None:
        return None


def _config(tmp_path: Path, **kwargs: object) -> WebChatMediaConfig:
    return WebChatMediaConfig(
        state_dir=tmp_path / "s",
        pending_dir=tmp_path / "p",
        events_dir=tmp_path / "e",
        store_dir=tmp_path / "w",
        **kwargs,
    )


def _png_size(data: bytes) -> tuple[int, int]:
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def test_manifest_uses_channel_slug_and_standalone(tmp_path: Path) -> None:
    config = _config(tmp_path, channel_name="Ops", channel_slug="ops-room")
    body = manifest(config)
    assert body["name"] == "Ops"
    assert body["start_url"] == "/c/ops-room"
    assert body["scope"] == "/"
    assert body["display"] == "standalone"
    icons = body["icons"]
    assert len(icons) == 2
    assert icons[0]["sizes"] == "192x192"


def test_manifest_and_sw_routes(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(tmp_path), _Bridge()))
    manifest_resp = client.get("/manifest.webmanifest")
    assert manifest_resp.status_code == 200
    assert manifest_resp.headers["content-type"].startswith("application/manifest+json")
    assert manifest_resp.json()["start_url"] == "/c/general"

    sw = client.get("/sw.js")
    assert sw.status_code == 200
    assert sw.headers.get("service-worker-allowed") == "/"
    assert sw.headers.get("cache-control") == "no-cache"
    assert "addEventListener('push'" in sw.text
    assert "fetch" not in sw.text


def test_default_pwa_icons_are_png_with_expected_size(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(tmp_path), _Bridge()))
    for path, size in (("/pwa/icon-192.png", 192), ("/pwa/icon-512.png", 512)):
        resp = client.get(path)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        w, h = _png_size(resp.content)
        assert (w, h) == (size, size)


def test_custom_pwa_icon_from_assets_dir(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    custom = assets / "mine-192.png"
    custom.write_bytes(resolve_pwa_icon(_config(tmp_path), 192).read_bytes())  # type: ignore[union-attr]
    config = _config(tmp_path, assets_dir=assets, pwa_icon_192="mine-192.png")
    client = TestClient(create_app(config, _Bridge()))
    assert client.get("/pwa/icon-192.png").content == custom.read_bytes()


def test_custom_icon_missing_or_unsafe_is_404(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    config = _config(tmp_path, assets_dir=assets, pwa_icon_192="missing.png")
    client = TestClient(create_app(config, _Bridge()))
    assert client.get("/pwa/icon-192.png").status_code == 404
    config2 = _config(tmp_path, assets_dir=assets, pwa_icon_192="../escape.png")
    client2 = TestClient(create_app(config2, _Bridge()))
    assert client2.get("/pwa/icon-192.png").status_code == 404


def test_index_html_pwa_contract() -> None:
    assert INDEX_HTML.isascii()
    assert 'rel="manifest"' in INDEX_HTML
    assert "/manifest.webmanifest" in INDEX_HTML
    assert "apple-mobile-web-app-capable" in INDEX_HTML
    assert 'register("/sw.js"' in INDEX_HTML
    assert "open-thread" in INDEX_HTML
    sw = service_worker_source()
    assert "showNotification" in sw
    assert "openWindow" in sw
    assert "postMessage" in sw
    assert "fetch" not in sw
    assert "caches" not in sw.lower()


def test_push_ui_strings_in_index_html() -> None:
    assert "PUSH_ENABLE" in INDEX_HTML
    assert "String.fromCharCode" in INDEX_HTML
    assert "push-btn" in INDEX_HTML
    assert "push-ios-hint" in INDEX_HTML
    assert "PushManager" in INDEX_HTML
