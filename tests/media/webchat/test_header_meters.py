"""Header usage meters contracts for WebChat (Issue #6022)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mltgnt.media.webchat.app import create_app  # noqa: E402
from mltgnt.media.webchat.config import WebChatMediaConfig  # noqa: E402
from mltgnt.media.webchat.ui import INDEX_HTML  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]


class _Bridge:
    def handle_event(self, _event: object) -> None:
        return None


def _base_dirs(tmp_path: Path) -> dict[str, Path]:
    return {
        "state_dir": tmp_path / "s",
        "pending_dir": tmp_path / "p",
        "events_dir": tmp_path / "e",
        "store_dir": tmp_path / "w",
    }


def test_header_meters_config_contract(tmp_path: Path) -> None:
    """Issue #6022: header_meters_url validation and GET /config."""
    dirs = _base_dirs(tmp_path)
    WebChatMediaConfig(**dirs, header_meters_url=None)
    cfg = WebChatMediaConfig(**dirs, header_meters_url="/meters.json")
    client = TestClient(create_app(cfg, _Bridge()))
    assert client.get("/config").json()["header_meters_url"] == "/meters.json"

    client_default = TestClient(create_app(WebChatMediaConfig(**dirs), _Bridge()))
    assert client_default.get("/config").json()["header_meters_url"] is None

    for bad in ("", "https://example.test/meters.json", "//example.test/meters.json", "meters.json"):
        with pytest.raises(ValueError, match="header_meters_url"):
            WebChatMediaConfig(**dirs, header_meters_url=bad)


def test_header_meters_disabled_contract() -> None:
    """Issue #6022: hidden region and no fetch when URL unset."""
    assert 'id="header-meters"' in INDEX_HTML
    assert "header-meters" in INDEX_HTML and "[hidden]" in INDEX_HTML or "hidden" in INDEX_HTML
    assert "function fetchHeaderMeters" in INDEX_HTML
    assert "function renderHeaderMeters" in INDEX_HTML
    assert "function meterView" in INDEX_HTML
    assert "headerMetersUrl" in INDEX_HTML
    assert "if (!headerMetersUrl)" in INDEX_HTML


def test_header_meters_render_contract() -> None:
    """Issue #6022: clamp, progress, footers, and failure hide."""
    assert "function meterView" in INDEX_HTML
    assert "Math.round" in INDEX_HTML and "clamp01" in INDEX_HTML
    assert "86400" in INDEX_HTML
    assert "window_seconds * 1000" in INDEX_HTML
    assert "1 - (resetsAtMs - nowMs)" in INDEX_HTML
    assert "0.1" in INDEX_HTML
    assert "String.fromCharCode" in INDEX_HTML
    assert "console.warn" in INDEX_HTML
    assert "hideHeaderMeters" in INDEX_HTML
    assert "textContent" in INDEX_HTML
    assert "innerHTML" not in INDEX_HTML.split("function renderHeaderMeters")[1].split("function fetchHeaderMeters")[0]
    assert "cache: \"no-store\"" in INDEX_HTML
    assert "validateHeaderMetersPayload" in INDEX_HTML


def test_header_meters_refresh_contract() -> None:
    """Issue #6022: fetch and tick intervals plus agent message refresh."""
    assert "HEADER_METERS_FETCH_MS = 300000" in INDEX_HTML
    assert "HEADER_METERS_TICK_MS = 60000" in INDEX_HTML
    assert "setInterval(fetchHeaderMeters" in INDEX_HTML
    assert "renderHeaderMeters(headerMetersPayload" in INDEX_HTML
    assert 'kind === "message"' in INDEX_HTML
    assert 'row.author !== "user"' in INDEX_HTML
    assert "fetchHeaderMeters()" in INDEX_HTML


def test_header_meters_responsive_contract() -> None:
    """Issue #6022: narrow layout and meter color variables."""
    assert "max-width: 479px" in INDEX_HTML
    assert "meter-name" in INDEX_HTML
    assert "meter-footer" in INDEX_HTML
    assert "--meter-fill" in INDEX_HTML
    assert "--meter-fill-warn" in INDEX_HTML
    assert "prefers-color-scheme: light" in INDEX_HTML
    assert "#header-meters" in INDEX_HTML
    assert "min-width: 0" in INDEX_HTML
    assert "white-space: nowrap" in INDEX_HTML


def test_header_meters_release_contract() -> None:
    """Issue #6022: README, CHANGELOG v0.135.0; pyproject version line present."""
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (_REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "header_meters_url" in readme
    assert "fetched_at" in readme
    assert '"groups"' in readme or "groups" in readme
    assert "## v0.135.0" in changelog
    assert re.search(r"header.?meter", changelog, re.I)
    assert re.search(r'^version = "\d+\.\d+\.\d+"', pyproject, re.M)
