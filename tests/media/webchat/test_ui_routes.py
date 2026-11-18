"""WebChat channel URL routes (#5613)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mltgnt.media._core.types import MediaEvent  # noqa: E402
from mltgnt.media.webchat.app import create_app  # noqa: E402
from mltgnt.media.webchat.config import WebChatMediaConfig  # noqa: E402
from mltgnt.media.webchat.ui import INDEX_HTML  # noqa: E402


class _Bridge:
    def handle_event(self, event: MediaEvent) -> None:
        return None


def _config(tmp_path: Path, **kwargs: object) -> WebChatMediaConfig:
    return WebChatMediaConfig(
        state_dir=tmp_path / "s",
        pending_dir=tmp_path / "p",
        events_dir=tmp_path / "e",
        store_dir=tmp_path / "w",
        **kwargs,
    )


def _client(tmp_path: Path, **kwargs: object) -> TestClient:
    return TestClient(create_app(_config(tmp_path, **kwargs), _Bridge()))


@pytest.mark.parametrize(
    "path",
    [
        "/c/general",
        "/c/general/t/abc123",
        "/c/general/p/abc123",
        "/c/general/bookmarks",
    ],
)
def test_channel_routes_serve_index_html(tmp_path: Path, path: str) -> None:
    resp = _client(tmp_path).get(path)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert resp.text == INDEX_HTML


def test_root_redirects_to_channel(tmp_path: Path) -> None:
    resp = _client(tmp_path).get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/c/general"


@pytest.mark.parametrize(
    ("path", "location"),
    [
        ("/c/other", "/c/general"),
        ("/c/other/t/abc123", "/c/general/t/abc123"),
    ],
)
def test_wrong_slug_redirects_to_configured_channel(tmp_path: Path, path: str, location: str) -> None:
    resp = _client(tmp_path).get(path, follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == location


def test_custom_channel_slug_redirect_and_config(tmp_path: Path) -> None:
    client = _client(tmp_path, channel_slug="ops-1")
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/c/ops-1"
    cfg = client.get("/config").json()
    assert cfg["default_channel"] == "ops-1"
    assert cfg["channels"] == [{"slug": "ops-1", "name": "general"}]


def test_index_html_uses_channel_api_and_channel_list() -> None:
    """Refs sumipan/nexus#7205 AC8."""
    assert INDEX_HTML.isascii()
    for token in ('"/api/c/"', "/api/channels", "channel-list", "webchat.read."):
        assert token in INDEX_HTML
    for forbidden in (
        'fetch("/messages',
        'fetch("/threads/',
        'fetch("/bookmarks',
        'new EventSource("/stream")',
        "config.channel_slug",
    ):
        assert forbidden not in INDEX_HTML



def _narrow_screen_css_block() -> str:
    """Extract the ``@media (max-width: 760px)`` rule block from ``INDEX_HTML``."""
    marker = "@media (max-width: 760px) {"
    start = INDEX_HTML.index(marker)
    depth = 0
    for i in range(start, len(INDEX_HTML)):
        ch = INDEX_HTML[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return INDEX_HTML[start : i + 1]
    raise AssertionError("unclosed @media (max-width: 760px) block")


def test_index_html_narrow_screen_switches_between_channel_list_and_main() -> None:
    """Refs sumipan/nexus#7239 AC1 and AC2."""
    block = _narrow_screen_css_block()
    for token in (
        "#app.show-channels #channel-list",
        "width: 100%",
        "#app.show-channels #main",
        "display: none",
        "#app.has-channels:not(.show-channels) #channel-list",
        "#app.has-channels #channel-back",
    ):
        assert token in block
    for token in (
        'id="channel-back"',
        "function syncChannelsView",
        '"#channels"',
        'addEventListener("hashchange"',
        "location.pathname + location.hash",
    ):
        assert token in INDEX_HTML


def test_index_html_marks_unread_channels_by_weight() -> None:
    """Refs sumipan/nexus#7239 AC4."""
    assert "#channel-list a.has-unread" in INDEX_HTML
    assert "#channel-back.has-unread" in INDEX_HTML
    assert "#channel-list .unread" not in INDEX_HTML
    assert 'dot.className = "unread"' not in INDEX_HTML


def test_index_html_prefers_row_display() -> None:
    """Issue #5956 AC6."""
    for token in (
        "function rowName(row)",
        "function rowAvatar(row, tiny)",
        "row.display",
        "/^https?:\\/\\//i",
        "rowAvatar(row, false)",
        "rowName(row)",
    ):
        assert token in INDEX_HTML
