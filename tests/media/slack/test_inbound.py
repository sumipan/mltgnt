"""mltgnt.media.slack.inbound (#4029)."""

from __future__ import annotations

from mltgnt.interfaces.turn import Attachment
from mltgnt.media.slack import inbound


def test_app_mention_strips_bot_mention() -> None:
    event = {"type": "app_mention", "channel": "C1", "user": "U9", "text": "<@U123> hello", "ts": "1.1"}
    got = inbound.to_media_event(event)
    assert got.text == "hello"
    assert got.space_id == "C1"
    assert got.conversation_id == "C1:1.1"
    assert got.message_id == "1.1"
    assert got.author == "U9"
    assert got.attachments == ()
    assert got.raw is event


def test_plain_message_keeps_mentions_unless_asked() -> None:
    event = {"type": "message", "channel": "C1", "text": " hi <@U1> ", "ts": "2.2", "thread_ts": "1.1"}
    assert inbound.to_media_event(event).text == "hi <@U1>"
    assert inbound.to_media_event(event, strip_bot_mentions=True).text == "hi"
    assert inbound.to_media_event(event).conversation_id == "C1:1.1"


def test_text_override_and_missing_fields() -> None:
    got = inbound.to_media_event({}, text_override="<@U1> x")
    assert got.text == "<@U1> x"
    assert got.conversation_id == ""
    assert got.space_id == ""


def test_file_share_becomes_attachment() -> None:
    event = {
        "type": "message",
        "subtype": "file_share",
        "channel": "C1",
        "user": "U9",
        "text": "see file",
        "ts": "3.3",
        "files": [
            {
                "id": "F1",
                "name": "photo.png",
                "mimetype": "image/png",
                "url_private_download": "https://files.example.test/F1/download",
                "url_private": "https://files.example.test/F1",
            }
        ],
    }
    got = inbound.to_media_event(event)
    assert got.attachments == (
        Attachment(name="photo.png", content_type="image/png", uri="https://files.example.test/F1/download"),
    )


def test_files_to_attachments_edge_cases() -> None:
    files = [
        "junk",
        {"id": "F2", "url_private": "https://files.example.test/F2"},
        {},
    ]
    assert inbound.files_to_attachments(files) == (
        Attachment(name="F2", content_type=None, uri="https://files.example.test/F2"),
        Attachment(name="file", content_type=None, uri=None),
    )
    assert inbound.files_to_attachments(None) == ()


def test_strip_mentions_and_thread_ts() -> None:
    assert inbound.strip_mentions("<@U1><@U2> go") == "go"
    assert inbound.strip_mentions("") == ""
    assert inbound.thread_ts_from_event({"ts": "1.0"}) == "1.0"
    assert inbound.thread_ts_from_event({"ts": "1.0", "thread_ts": "0.5"}) == "0.5"
    assert inbound.thread_ts_from_event({}) == ""
