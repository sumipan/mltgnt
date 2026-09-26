"""MediaEvent / OutboundMessage / MediaConfig (#4027)."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from mltgnt.config.language import JA
from mltgnt.interfaces.turn import Attachment
from mltgnt.media._core.client import MediaClient, Status, adapt_client
from mltgnt.media._core.config import MediaConfig
from mltgnt.media._core.types import MediaEvent, OutboundMessage


def test_media_event_fields_and_immutability() -> None:
    att = Attachment(name="a.txt", content_type="text/plain")
    event = MediaEvent(
        space_id="S1",
        conversation_id="C1",
        message_id="M1",
        author="U1",
        text="hello",
        attachments=(att,),
        raw={"k": "v"},
    )
    assert event.attachments == (att,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.text = "x"  # type: ignore[misc]


def test_media_event_defaults() -> None:
    event = MediaEvent(space_id="S1", conversation_id="C1", message_id="M1", author="U1", text="hi")
    assert event.attachments == ()
    assert event.raw is None


def test_outbound_message() -> None:
    msg = OutboundMessage(text="hi", space_id="S1")
    assert msg.thread_id is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        msg.text = "x"  # type: ignore[misc]


def test_media_config_requires_paths(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        MediaConfig()  # type: ignore[call-arg]
    cfg = MediaConfig(state_dir=tmp_path / "s", pending_dir=tmp_path / "p", events_dir=tmp_path / "e")
    assert cfg.language is JA
    assert cfg.progress_min_interval_sec > 0
    assert cfg.approval_ttl_sec > 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.state_dir = tmp_path  # type: ignore[misc]


def test_media_config_is_subclassable(tmp_path: Path) -> None:
    @dataclasses.dataclass(frozen=True)
    class SubConfig(MediaConfig):
        token_env: str = "TOKEN"

    cfg = SubConfig(state_dir=tmp_path, pending_dir=tmp_path, events_dir=tmp_path)
    assert cfg.token_env == "TOKEN"


def test_client_reexports() -> None:
    from mltgnt.interfaces import media

    assert MediaClient is media.MediaClient
    assert Status is media.Status
    assert adapt_client is media.adapt_client


def test_media_packages_export_nothing() -> None:
    import mltgnt.media
    import mltgnt.media._core

    assert mltgnt.media.__all__ == []
    assert mltgnt.media._core.__all__ == []


def test_language_pack_media_vocabulary() -> None:
    assert JA.approval_words
    assert set(JA.status_labels) == {s.value for s in Status}
    assert JA.enqueue_failed_text
    assert JA.progress_line_pattern.pattern
