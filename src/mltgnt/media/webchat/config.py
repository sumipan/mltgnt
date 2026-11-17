"""WebChat-specific media settings."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from mltgnt.media._core.config import MediaConfig

__all__ = [
    "ResolvedWebChatChannel",
    "WebChatChannel",
    "WebChatMediaConfig",
    "WebPushConfig",
    "validate_header_meters_url",
]


def validate_header_meters_url(value: str | None) -> str | None:
    """Return a same-origin absolute path, or ``None`` when unset."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("header_meters_url must be a same-origin absolute path starting with /")
    # WHATWG URL parsing treats backslashes as slashes and strips ASCII tabs
    # and newlines.  Reject those forms before the browser can normalize a
    # seemingly relative path into a protocol-relative, cross-origin URL.
    if (
        not value.startswith("/")
        or value.startswith("//")
        or "\\" in value
        or any(char in value for char in ("\t", "\n", "\r"))
    ):
        raise ValueError("header_meters_url must be a same-origin absolute path starting with /")
    return value


def _default_push_body_text() -> str:
    return "".join(map(chr, (0x8FD4, 0x4FE1, 0x304C, 0x3042, 0x308A, 0x307E, 0x3057, 0x305F)))


@dataclass(frozen=True)
class WebPushConfig:
    """VAPID keys and notification copy for Web Push (optional ``webchat-push`` extra)."""

    enabled: bool = False
    vapid_public_key: str | None = None
    vapid_private_key: str | None = None
    vapid_subject: str = "mailto:admin@example.invalid"
    body_text: str = field(default_factory=_default_push_body_text)


@dataclass(frozen=True)
class WebChatChannel:
    """One additional WebChat channel declared in ``WebChatMediaConfig.channels``."""

    slug: str
    name: str
    author: str | None = None
    store_dir: Path | None = None


@dataclass(frozen=True)
class ResolvedWebChatChannel:
    """A WebChat channel after defaults are applied (default channel or an entry in ``channels``)."""

    slug: str
    name: str
    space_id: str
    store_dir: Path
    author: str | None


@dataclass(frozen=True)
class WebChatMediaConfig(MediaConfig):
    """Message store directory (required) and the local listen address of the single channel.

    ``display_names`` and ``avatars`` are keyed by message author (the inbound user is
    ``"user"``; a persona posts as its client ``author``). An avatar is a file name under
    ``assets_dir`` or an ``http(s)://`` URL. ``channel_name`` is shown in the UI header and composer. Message images live in
    ``files_dir`` (default: ``<store_dir>/../files``), at most ``max_upload_bytes`` each.
    """

    store_dir: Path = field(kw_only=True)
    host: str = "127.0.0.1"
    port: int = 8765
    space_id: str = "webchat"
    assets_dir: Path | None = None
    avatars: Mapping[str, str] = field(default_factory=dict, hash=False)
    display_names: Mapping[str, str] = field(default_factory=dict, hash=False)
    channel_name: str = "general"
    channel_slug: str = "general"
    files_dir: Path | None = None
    max_upload_bytes: int = 20 * 1024 * 1024
    push: WebPushConfig = field(default_factory=WebPushConfig)
    pwa_icon_192: str | None = None
    pwa_icon_512: str | None = None
    header_meters_url: str | None = None
    channels: tuple[WebChatChannel, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "header_meters_url", validate_header_meters_url(self.header_meters_url))
        if not re.fullmatch(r"[a-z0-9-]+", self.channel_slug):
            raise ValueError("channel_slug must match [a-z0-9-]+")
        if not isinstance(self.avatars, MappingProxyType):
            object.__setattr__(self, "avatars", MappingProxyType(dict(self.avatars)))
        if not isinstance(self.display_names, MappingProxyType):
            object.__setattr__(self, "display_names", MappingProxyType(dict(self.display_names)))
        ch_tuple = tuple(self.channels)
        object.__setattr__(self, "channels", ch_tuple)
        seen: set[str] = set()
        for ch in ch_tuple:
            if not re.fullmatch(r"[a-z0-9-]+", ch.slug):
                raise ValueError("channel slug must match [a-z0-9-]+")
            if ch.slug == self.channel_slug:
                raise ValueError("channel slug must not match channel_slug")
            if ch.slug in seen:
                raise ValueError("duplicate channel slug")
            seen.add(ch.slug)

    @property
    def resolved_channels(self) -> tuple[ResolvedWebChatChannel, ...]:
        default = ResolvedWebChatChannel(
            slug=self.channel_slug,
            name=self.channel_name,
            space_id=self.space_id,
            store_dir=Path(self.store_dir),
            author=None,
        )
        parent = Path(self.store_dir).parent
        extra: list[ResolvedWebChatChannel] = []
        for ch in self.channels:
            store_dir = ch.store_dir if ch.store_dir is not None else parent / "channels" / ch.slug
            extra.append(
                ResolvedWebChatChannel(
                    slug=ch.slug,
                    name=ch.name,
                    space_id=f"{self.space_id}.{ch.slug}",
                    store_dir=Path(store_dir),
                    author=ch.author,
                )
            )
        return (default, *extra)

    def channel_by_space(self, space: str) -> ResolvedWebChatChannel | None:
        for ch in self.resolved_channels:
            if ch.space_id == space:
                return ch
        return None

    def channel_by_slug(self, slug: str) -> ResolvedWebChatChannel | None:
        for ch in self.resolved_channels:
            if ch.slug == slug:
                return ch
        return None

    @property
    def resolved_files_dir(self) -> Path:
        """``files_dir``, or ``files`` next to ``store_dir`` when unset."""
        return self.files_dir if self.files_dir is not None else Path(self.store_dir).parent / "files"
