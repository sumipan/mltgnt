"""PWA manifest, service worker source, and icon resolution for WebChat."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mltgnt.media.webchat.config import WebChatMediaConfig

__all__ = ["manifest", "resolve_pwa_icon", "service_worker_source"]

_PWA_ICON_FIELD = {192: "pwa_icon_192", 512: "pwa_icon_512"}


def _safe_asset_path(assets_dir: Path, name: str) -> Path | None:
    if not name or "/" in name or "\\" in name or name in (".", "..") or ".." in name:
        return None
    base = assets_dir.resolve()
    candidate = (assets_dir / name).resolve()
    if candidate.parent != base or not candidate.is_file():
        return None
    return candidate


def manifest(config: WebChatMediaConfig) -> dict[str, object]:
    """Web app manifest dict for ``GET /manifest.webmanifest``."""
    slug = config.channel_slug
    start = f"/c/{slug}"
    return {
        "name": config.channel_name,
        "short_name": config.channel_name,
        "start_url": start,
        "scope": "/",
        "display": "standalone",
        "theme_color": "#1a1d21",
        "background_color": "#1a1d21",
        "icons": [
            {"src": "/pwa/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/pwa/icon-512.png", "sizes": "512x512", "type": "image/png"},
        ],
    }


def service_worker_source() -> str:
    """Minimal service worker handling push and notification clicks only (ASCII)."""
    return (
        "self.addEventListener('push', function (event) {\n"
        "  if (!event.data) { return; }\n"
        "  var payload;\n"
        "  try { payload = event.data.json(); } catch (e) { return; }\n"
        "  var title = payload.title || '';\n"
        "  var body = payload.body || '';\n"
        "  var url = payload.url || '/';\n"
        "  var tag = payload.tag || 'webchat';\n"
        "  event.waitUntil(self.registration.showNotification(title, {\n"
        "    body: body,\n"
        "    tag: tag,\n"
        "    data: { url: url }\n"
        "  }));\n"
        "});\n"
        "self.addEventListener('notificationclick', function (event) {\n"
        "  event.notification.close();\n"
        "  var url = (event.notification.data && event.notification.data.url) || '/';\n"
        "  event.waitUntil(clients.matchAll({ type: 'window', includeUncontrolled: true })\n"
        "    .then(function (list) {\n"
        "      for (var i = 0; i < list.length; i++) {\n"
        "        var client = list[i];\n"
        "        if ('focus' in client) {\n"
        "          client.postMessage({ type: 'open-thread', url: url });\n"
        "          return client.focus();\n"
        "        }\n"
        "      }\n"
        "      if (clients.openWindow) { return clients.openWindow(url); }\n"
        "    }));\n"
        "});\n"
    )


def _builtin_icon_path(size: Literal[192, 512]) -> Path:
    return Path(__file__).resolve().parent / "assets" / f"pwa-icon-{size}.png"


def resolve_pwa_icon(config: WebChatMediaConfig, size: Literal[192, 512]) -> Path | None:
    """Return a readable icon path, or None when the configured asset is missing or unsafe."""
    field = _PWA_ICON_FIELD[size]
    custom_name = getattr(config, field, None)
    if custom_name:
        if config.assets_dir is None:
            return None
        path = _safe_asset_path(config.assets_dir, custom_name)
        return path
    builtin = _builtin_icon_path(size)
    return builtin if builtin.is_file() else None
