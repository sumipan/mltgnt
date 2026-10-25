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
    assert client.get("/config").json()["channel_slug"] == "ops-1"


def test_index_html_routing_client_strings() -> None:
    forbidden = [
        'fetch("config"',
        'fetch("messages',
        'fetch("threads/',
        'fetch("bookmarks"',
        'fetch("files',
        'new EventSource("stream"',
        '"files/"',
        '"assets/"',
    ]
    for token in forbidden:
        assert token not in INDEX_HTML
    for token in (
        'fetch("/config"',
        'fetch("/messages',
        'fetch("/threads/',
        'fetch("/bookmarks"',
        'fetch("/files',
        'new EventSource("/stream"',
        '"/files/"',
        '"/assets/"',
        "history.pushState",
        "history.replaceState",
        "popstate",
        "navigator.clipboard",
    ):
        assert token in INDEX_HTML
