"""mltgnt.media.webchat.config (#7201): multi-channel resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from mltgnt.media.webchat.config import WebChatChannel, WebChatMediaConfig


def _root(tmp_path: Path) -> Path:
    return tmp_path / "root"


def _multi_channel_config(tmp_path: Path) -> WebChatMediaConfig:
    root = _root(tmp_path)
    return WebChatMediaConfig(
        state_dir=root / "s",
        pending_dir=root / "p",
        events_dir=root / "e",
        store_dir=root / "chat" / "channel",
        channel_slug="main",
        channel_name="Main",
        channels=(WebChatChannel(slug="side", name="Side", author="persona-a"),),
    )


def test_resolved_channels_default_first_then_declared(tmp_path: Path) -> None:
    """Issue #7201 AC1."""
    config = _multi_channel_config(tmp_path)
    root = _root(tmp_path)
    resolved = config.resolved_channels
    assert len(resolved) == 2
    default_ch, side_ch = resolved
    assert (default_ch.slug, default_ch.name, default_ch.space_id, default_ch.store_dir, default_ch.author) == (
        "main",
        "Main",
        "webchat",
        root / "chat" / "channel",
        None,
    )
    assert (side_ch.slug, side_ch.name, side_ch.space_id, side_ch.store_dir, side_ch.author) == (
        "side",
        "Side",
        "webchat.side",
        root / "chat" / "channels" / "side",
        "persona-a",
    )
    assert config.channel_by_space("webchat.side") == side_ch
    assert config.channel_by_slug("side") == side_ch
    assert config.channel_by_space("other") is None


@pytest.mark.parametrize(
    "channels",
    [
        (WebChatChannel(slug="Bad Slug", name="x"),),
        (
            WebChatChannel(slug="dup", name="a"),
            WebChatChannel(slug="dup", name="b"),
        ),
        (WebChatChannel(slug="main", name="conflict"),),
    ],
)
def test_channels_reject_bad_duplicate_or_default_slug(
    tmp_path: Path, channels: tuple[WebChatChannel, ...]
) -> None:
    """Issue #7201 AC2."""
    root = _root(tmp_path)
    with pytest.raises(ValueError):
        WebChatMediaConfig(
            state_dir=root / "s",
            pending_dir=root / "p",
            events_dir=root / "e",
            store_dir=root / "chat" / "channel",
            channel_slug="main",
            channel_name="Main",
            channels=channels,
        )
